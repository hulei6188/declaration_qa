"""申报问答 LangGraph 工作流和业务节点实现。"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from collections.abc import Callable, Iterable

from declaration_qa.audit import InMemoryAuditStore
from declaration_qa.business import BusinessDataAdapter, InMemoryBusinessDataAdapter
from declaration_qa.evidence import decide_evidence
from declaration_qa.prompting import DeterministicAnswerGenerator
from declaration_qa.question_planner import RuleQuestionPlanner
from declaration_qa.rag_engine import InMemoryRAGEngine, build_seed_rag_engine, normalize_text
from declaration_qa.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, TICKET_AUGMENTED_CHAIN
from declaration_qa.router import (
    INTENT_REJECTION_REASON,
    INTENT_TICKET_STATUS,
    ROUTE_CLARIFICATION,
    ROUTE_REFUSE,
    ROUTE_TICKET_CONTEXT_RELATED,
    ROUTE_TICKET_CONTEXT_UNRELATED,
)
from declaration_qa.schemas import ChatRequest, ChatResponse
from declaration_qa.state import DeclarationQAState


class DeclarationQAWorkflow:
    """申报问答的图工作流编排器。"""
    def __init__(
        self,
        *,
        rag_engine: InMemoryRAGEngine | None = None,
        business_adapter: BusinessDataAdapter | None = None,
        audit_store: InMemoryAuditStore | None = None,
        answer_generator: DeterministicAnswerGenerator | None = None,
        question_planner: RuleQuestionPlanner | None = None,
    ) -> None:
        """初始化申报问答工作流。"""
        self.rag_engine = rag_engine or build_seed_rag_engine()
        self.business_adapter = business_adapter or InMemoryBusinessDataAdapter()
        self.audit_store = audit_store or InMemoryAuditStore()
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.question_planner = question_planner or RuleQuestionPlanner()
        self.graph = self._build_graph()

    def invoke(self, request: ChatRequest) -> ChatResponse:
        """同步执行问答工作流并返回最终响应。"""
        initial_state = self._initial_state(request)
        final_state = self.graph.invoke(initial_state)
        return ChatResponse(
            session_id=str(final_state["session_id"]),
            audit_id=str(final_state["audit_id"]),
            intent=str(final_state.get("intent") or "unknown"),
            route=str(final_state.get("route") or ""),
            sub_intent=str(final_state.get("sub_intent") or ""),
            route_decision=final_state.get("route_decision") or {},
            multi_question=bool(final_state.get("multi_question")),
            question_plan=final_state.get("question_plan") or {},
            sub_tasks=final_state.get("sub_tasks") or [],
            need_ticket_selection=bool(final_state.get("need_ticket_selection")),
            ticket_selection_reason=str(final_state.get("ticket_selection_reason") or ""),
            tickets=final_state.get("tickets") or [],
            ticket_context=final_state.get("ticket_context") or {},
            retrieval_chain=str(final_state.get("retrieval_chain") or ""),
            answer=str(final_state.get("answer") or ""),
            refused=bool(final_state.get("refused")),
            refuse_reason=final_state.get("refuse_reason"),
            sources=final_state.get("retrieved_sources") or [],
            business_snapshot=final_state.get("business_snapshot") or {},
            rag_status=final_state.get("rag_status") or {},
            evidence_decision=final_state.get("evidence_decision") or {},
            suggested_actions=final_state.get("suggested_actions") or [],
            events=final_state.get("events") or [],
        )

    def stream(self, request: ChatRequest) -> Iterable[dict[str, Any]]:
        """流式执行问答工作流并产出事件。"""
        state: DeclarationQAState = self._initial_state(request)
        yield {"event": "run_started", "data": {"session_id": state["session_id"]}}

        def run_node(name: str, handler) -> None:
            """执行工作流节点并记录节点事件。"""
            nonlocal state
            before = len(state.get("events") or [])
            state = handler(state)
            for event in (state.get("events") or [])[before:]:
                yield_event = {"event": event.get("event") or name, "data": event.get("data") or {}}
                emitted.append(yield_event)

        emitted: list[dict[str, Any]] = []

        def drain() -> Iterable[dict[str, Any]]:
            """收集流式文本片段并返回完整文本。"""
            while emitted:
                yield emitted.pop(0)

        for name, handler in [
            ("context_build", self.context_build),
            ("query_preprocess", self.query_preprocess),
            ("question_plan", self.question_plan),
        ]:
            run_node(name, handler)
            yield from drain()

        if state.get("multi_question"):
            run_node("multi_question_collect", self.multi_question_collect)
            yield from drain()
            if state.get("need_ticket_selection"):
                run_node("refuse_or_human", self.refuse_or_human)
                yield from drain()
                yield from token_events(str(state.get("answer") or ""))
            else:
                if state.get("retrieved_sources"):
                    yield {"event": "sources", "data": {"items": state.get("retrieved_sources") or []}}
                yield from self.llm_generate_stream_events(state)
            for name, handler in [
                ("answer_postprocess", self.answer_postprocess),
                ("audit_log", self.audit_log),
            ]:
                run_node(name, handler)
                yield from drain()
            yield {
                "event": "done",
                "data": {
                    "session_id": state.get("session_id"),
                    "audit_id": state.get("audit_id"),
                    "intent": state.get("intent"),
                    "route": state.get("route"),
                    "sub_intent": state.get("sub_intent"),
                    "multi_question": bool(state.get("multi_question")),
                    "question_plan": state.get("question_plan") or {},
                    "sub_tasks": state.get("sub_tasks") or [],
                    "need_ticket_selection": bool(state.get("need_ticket_selection")),
                    "ticket_selection_reason": state.get("ticket_selection_reason"),
                    "tickets": state.get("tickets") or [],
                    "ticket_context": state.get("ticket_context") or {},
                    "retrieval_chain": state.get("retrieval_chain"),
                    "refused": bool(state.get("refused")),
                    "refuse_reason": state.get("refuse_reason"),
                    "suggested_actions": state.get("suggested_actions") or [],
                },
            }
            return

        for name, handler in [
            ("intent_route", self.intent_route),
        ]:
            run_node(name, handler)
            yield from drain()

        route = state.get("route") or ROUTE_REFUSE
        if route == ROUTE_TICKET_CONTEXT_RELATED:
            for name, handler in [
                ("ticket_list_query", self.ticket_list_query),
                ("ticket_selection", self.ticket_selection),
                ("ticket_detail_query", self.ticket_detail_query),
                ("ticket_context_build", self.ticket_context_build),
                ("ticket_knowledge_rag", self.ticket_knowledge_rag),
                ("evidence_check", self.evidence_check),
            ]:
                run_node(name, handler)
                yield from drain()
                if state.get("refuse_reason") and name in {"ticket_list_query", "ticket_selection", "ticket_detail_query"}:
                    break
                if name == "ticket_knowledge_rag" and state.get("retrieved_sources"):
                    yield {"event": "sources", "data": {"items": state.get("retrieved_sources") or []}}
            if state.get("refuse_reason"):
                run_node("refuse_or_human", self.refuse_or_human)
                yield from drain()
                yield from token_events(str(state.get("answer") or ""))
            elif state.get("evidence_ok"):
                yield from self.llm_generate_stream_events(state)
            else:
                run_node("refuse_or_human", self.refuse_or_human)
                yield from drain()
                yield from token_events(str(state.get("answer") or ""))
        elif route == ROUTE_TICKET_CONTEXT_UNRELATED:
            for name, handler in [
                ("knowledge_filter_build", self.knowledge_filter_build),
                ("hybrid_retrieve", self.hybrid_retrieve),
                ("evidence_check", self.evidence_check),
            ]:
                run_node(name, handler)
                yield from drain()
                if name == "hybrid_retrieve" and state.get("retrieved_sources"):
                    yield {"event": "sources", "data": {"items": state.get("retrieved_sources") or []}}
            if state.get("evidence_ok"):
                yield from self.llm_generate_stream_events(state)
            else:
                run_node("refuse_or_human", self.refuse_or_human)
                yield from drain()
                yield from token_events(str(state.get("answer") or ""))
        else:
            run_node("refuse_or_human", self.refuse_or_human)
            yield from drain()
            yield from token_events(str(state.get("answer") or ""))

        for name, handler in [
            ("answer_postprocess", self.answer_postprocess),
            ("audit_log", self.audit_log),
        ]:
            run_node(name, handler)
            yield from drain()

        yield {
            "event": "done",
            "data": {
                "session_id": state.get("session_id"),
                "audit_id": state.get("audit_id"),
                "intent": state.get("intent"),
                "route": state.get("route"),
                "sub_intent": state.get("sub_intent"),
                "multi_question": bool(state.get("multi_question")),
                "question_plan": state.get("question_plan") or {},
                "sub_tasks": state.get("sub_tasks") or [],
                "need_ticket_selection": bool(state.get("need_ticket_selection")),
                "ticket_selection_reason": state.get("ticket_selection_reason"),
                "tickets": state.get("tickets") or [],
                "ticket_context": state.get("ticket_context") or {},
                "retrieval_chain": state.get("retrieval_chain"),
                "refused": bool(state.get("refused")),
                "refuse_reason": state.get("refuse_reason"),
                "suggested_actions": state.get("suggested_actions") or [],
            },
        }

    def llm_generate_stream_events(self, state: DeclarationQAState) -> Iterable[dict[str, Any]]:
        """流式调用回答生成器并产出文本事件。"""
        stream_generate = getattr(self.answer_generator, "stream_generate", None)
        if not callable(stream_generate):
            before = len(state.get("events") or [])
            next_state = self.llm_generate(state)
            state.clear()
            state.update(next_state)
            for event in (state.get("events") or [])[before:]:
                yield {"event": event.get("event") or "answer_generated", "data": event.get("data") or {}}
            yield from token_events(str(state.get("answer") or ""))
            return

        token_stream, prompt = stream_generate(dict(state))
        answer_parts = []
        for token in token_stream:
            text = str(token or "")
            if not text:
                continue
            answer_parts.append(text)
            yield {"event": "token", "data": {"content": text}}
        state["answer"] = "".join(answer_parts)
        audit_payload = dict(state.get("audit_payload") or {})
        audit_payload["final_prompt"] = prompt
        state["audit_payload"] = audit_payload
        add_event(state, "answer_generated", {"intent": state.get("intent"), "streamed": True})
        yield {"event": "answer_generated", "data": {"intent": state.get("intent"), "streamed": True}}

    def _initial_state(self, request: ChatRequest) -> DeclarationQAState:
        """根据请求构造工作流初始状态。"""
        apply_context = request.apply_context.model_dump(exclude_none=True) if request.apply_context else None
        return {
            "user_id": request.user_id,
            "session_id": request.session_id or uuid4().hex,
            "query": request.query,
            "normalized_query": "",
            "region_code": request.region_code,
            "app_version": request.app_version,
            "policy_version": request.policy_version,
            "biz_type": request.biz_type,
            "apply_id": request.apply_id,
            "apply_context": apply_context,
            "metadata_filter": request.metadata_filter,
            "intent": "unknown",
            "sub_intent": "",
            "route": "",
            "route_decision": {},
            "question_plan": {},
            "requires_ticket_context": False,
            "multi_question": False,
            "sub_tasks": [],
            "retrieval_chain": "",
            "need_ticket_selection": False,
            "ticket_selection_reason": "",
            "tickets": [],
            "ticket_context": None,
            "rag_query": "",
            "business_snapshot": None,
            "retrieval_filter": {},
            "retrieved_sources": [],
            "rag_status": {},
            "evidence_ok": False,
            "evidence_decision": {},
            "refuse_reason": None,
            "answer": "",
            "refused": False,
            "suggested_actions": [],
            "events": [],
            "audit_payload": {},
        }

    def _build_graph(self):
        """构建 LangGraph 节点和边。"""
        builder = StateGraph(DeclarationQAState)
        builder.add_node("context_build", self.context_build)
        builder.add_node("query_preprocess", self.query_preprocess)
        builder.add_node("question_plan", self.question_plan)
        builder.add_node("multi_question_collect", self.multi_question_collect)
        builder.add_node("intent_route", self.intent_route)
        builder.add_node("ticket_list_query", self.ticket_list_query)
        builder.add_node("ticket_selection", self.ticket_selection)
        builder.add_node("ticket_detail_query", self.ticket_detail_query)
        builder.add_node("ticket_context_build", self.ticket_context_build)
        builder.add_node("ticket_knowledge_rag", self.ticket_knowledge_rag)
        builder.add_node("knowledge_filter_build", self.knowledge_filter_build)
        builder.add_node("hybrid_retrieve", self.hybrid_retrieve)
        builder.add_node("evidence_check", self.evidence_check)
        builder.add_node("refuse_or_human", self.refuse_or_human)
        builder.add_node("llm_generate", self.llm_generate)
        builder.add_node("answer_postprocess", self.answer_postprocess)
        builder.add_node("audit_log", self.audit_log)

        builder.add_edge(START, "context_build")
        builder.add_edge("context_build", "query_preprocess")
        builder.add_edge("query_preprocess", "question_plan")
        builder.add_conditional_edges(
            "question_plan",
            lambda state: "multi" if state.get("multi_question") else "single",
            {"multi": "multi_question_collect", "single": "intent_route"},
        )
        builder.add_conditional_edges(
            "multi_question_collect",
            lambda state: "needs_selection" if state.get("need_ticket_selection") else "ready",
            {"needs_selection": "refuse_or_human", "ready": "llm_generate"},
        )
        builder.add_conditional_edges(
            "intent_route",
            lambda state: state.get("route") or "refuse",
            {
                ROUTE_TICKET_CONTEXT_RELATED: "ticket_list_query",
                ROUTE_TICKET_CONTEXT_UNRELATED: "knowledge_filter_build",
                ROUTE_CLARIFICATION: "refuse_or_human",
                ROUTE_REFUSE: "refuse_or_human",
            },
        )
        builder.add_conditional_edges(
            "ticket_list_query",
            lambda state: "missing" if state.get("refuse_reason") else "ok",
            {"ok": "ticket_selection", "missing": "refuse_or_human"},
        )
        builder.add_conditional_edges(
            "ticket_selection",
            lambda state: "selected" if state.get("apply_context") and not state.get("need_ticket_selection") else "needs_selection",
            {"selected": "ticket_detail_query", "needs_selection": "refuse_or_human"},
        )
        builder.add_conditional_edges(
            "ticket_detail_query",
            lambda state: "missing" if state.get("refuse_reason") else "ok",
            {"ok": "ticket_context_build", "missing": "refuse_or_human"},
        )
        builder.add_edge("ticket_context_build", "ticket_knowledge_rag")
        builder.add_edge("ticket_knowledge_rag", "evidence_check")
        builder.add_edge("knowledge_filter_build", "hybrid_retrieve")
        builder.add_edge("hybrid_retrieve", "evidence_check")
        builder.add_conditional_edges(
            "evidence_check",
            lambda state: "ok" if state.get("evidence_ok") else "refuse",
            {"ok": "llm_generate", "refuse": "refuse_or_human"},
        )
        builder.add_edge("llm_generate", "answer_postprocess")
        builder.add_edge("refuse_or_human", "answer_postprocess")
        builder.add_edge("answer_postprocess", "audit_log")
        builder.add_edge("audit_log", END)
        return builder.compile()

    def context_build(self, state: DeclarationQAState) -> DeclarationQAState:
        """构造基础请求上下文。"""
        next_state = dict(state)
        if not next_state.get("apply_context") and next_state.get("apply_id"):
            next_state["apply_context"] = self.business_adapter.get_apply_context(
                user_id=int(next_state["user_id"]),
                apply_id=str(next_state["apply_id"]),
            )
        if next_state.get("apply_context"):
            context_apply_id = str((next_state.get("apply_context") or {}).get("apply_id") or next_state.get("apply_id") or "")
            if context_apply_id and not self.business_adapter.verify_apply_access(
                user_id=int(next_state["user_id"]),
                apply_id=context_apply_id,
            ):
                next_state["apply_context"] = None
                next_state["refuse_reason"] = "apply_access_denied"
                add_event(next_state, "apply_access_denied", {"apply_id": context_apply_id})
        apply_context = next_state.get("apply_context") or {}
        next_state["biz_type"] = next_state.get("biz_type") or apply_context.get("biz_type")
        next_state["region_code"] = apply_context.get("region_code") or next_state.get("region_code") or "340000"
        add_event(next_state, "context_built", {"has_apply_context": bool(next_state.get("apply_context"))})
        return next_state

    def query_preprocess(self, state: DeclarationQAState) -> DeclarationQAState:
        """清理用户问题并写入状态。"""
        next_state = dict(state)
        query = normalize_text(mask_sensitive_text(next_state.get("query") or ""))
        next_state["normalized_query"] = query
        next_state["biz_type"] = next_state.get("biz_type") or infer_biz_type(query)
        add_event(next_state, "query_preprocessed", {"normalized_query": query, "biz_type": next_state.get("biz_type")})
        return next_state

    def question_plan(self, state: DeclarationQAState) -> DeclarationQAState:
        """拆解多问题请求。"""
        next_state = dict(state)
        plan = self.question_planner.plan(next_state)
        questions = []
        for index, item in enumerate(plan.questions, start=1):
            question = dict(item)
            question["id"] = question.get("id") or f"q{index}"
            questions.append(question)
        next_state["question_plan"] = {**plan.payload(), "questions": questions}
        next_state["multi_question"] = bool(plan.is_multi_question and len(questions) > 1)
        next_state["sub_tasks"] = questions if next_state["multi_question"] else []
        if not next_state["multi_question"] and questions:
            apply_planned_question(next_state, questions[0])
        add_event(
            next_state,
            "question_plan",
            {
                "multi_question": next_state["multi_question"],
                "count": len(questions),
                "provider": plan.provider,
                "reason": plan.reason,
                "expanded": any(bool(question.get("is_expanded")) for question in questions),
            },
        )
        return next_state

    def multi_question_collect(self, state: DeclarationQAState) -> DeclarationQAState:
        """对每个子问题单独路由、查询业务数据和检索证据。"""
        next_state = dict(state)
        sub_tasks = []
        all_sources = []
        tickets = []
        route_values = []
        intent_values = []
        rag_queries = []
        need_ticket_selection = False
        for index, question in enumerate(next_state.get("sub_tasks") or [], start=1):
            task_state = self.state_for_sub_question(next_state, question)
            task = self.collect_sub_question(task_state, question, index)
            sub_tasks.append(task)
            all_sources.extend(task.get("sources") or [])
            if task.get("need_ticket_selection"):
                need_ticket_selection = True
                tickets.extend(task.get("tickets") or [])
            if task.get("route"):
                route_values.append(task["route"])
            if task.get("intent"):
                intent_values.append(task["intent"])
            if task.get("rag_query"):
                rag_queries.append(str(task.get("rag_query") or ""))
        next_state["need_ticket_selection"] = need_ticket_selection
        ticket_selection_reason = next(
            (
                str(task.get("ticket_selection_reason") or "")
                for task in sub_tasks
                if task.get("need_ticket_selection") and task.get("ticket_selection_reason")
            ),
            "",
        )
        next_state["ticket_selection_reason"] = ticket_selection_reason or ("multiple_candidates" if need_ticket_selection else "")
        next_state["tickets"] = unique_tickets(tickets)
        next_state["sub_tasks"] = sub_tasks
        next_state["retrieved_sources"] = [] if need_ticket_selection else unique_sources(all_sources)
        next_state["route"] = "multi_question"
        next_state["intent"] = "multi_question"
        next_state["sub_intent"] = ",".join(unique_strings(intent_values))
        next_state["retrieval_chain"] = "multi_question_grouped_evidence"
        next_state["rag_query"] = "\n".join(unique_strings(rag_queries))
        next_state["route_decision"] = {"route": "multi_question", "sub_routes": unique_strings(route_values)}
        next_state["evidence_ok"] = False if need_ticket_selection else any(task.get("status") in {"ready", "answered"} for task in sub_tasks)
        if need_ticket_selection:
            next_state["refuse_reason"] = "ticket_selection_required"
        next_state["evidence_decision"] = {
            "ok": next_state["evidence_ok"],
            "reason": (
                "ticket_selection_required"
                if need_ticket_selection
                else ("" if next_state["evidence_ok"] else "no_sub_question_evidence")
            ),
            "sub_task_count": len(sub_tasks),
            "ready_count": sum(1 for task in sub_tasks if task.get("status") in {"ready", "answered"}),
        }
        add_event(
            next_state,
            "multi_question_collected",
            {
                "count": len(sub_tasks),
                "need_ticket_selection": need_ticket_selection,
                "source_count": len(next_state["retrieved_sources"]),
            },
        )
        return next_state

    def state_for_sub_question(self, state: DeclarationQAState, question: dict[str, Any]) -> DeclarationQAState:
        """构造子问题状态，复用原始请求上下文。"""
        query = normalize_text(str(question.get("query") or ""))
        inherited_apply_context = state.get("apply_context")
        inherited_biz_type = (
            (inherited_apply_context or {}).get("biz_type")
            if inherited_apply_context
            else infer_biz_type(query)
        )
        return {
            **state,
            "query": query,
            "normalized_query": query,
            "requires_ticket_context": bool(question.get("requires_ticket_context")),
            "biz_type": inherited_biz_type,
            "multi_question": False,
            "sub_tasks": [],
            "question_plan": {},
            "intent": "unknown",
            "sub_intent": "",
            "route": "",
            "route_decision": {},
            "retrieval_chain": "",
            "need_ticket_selection": False,
            "ticket_selection_reason": "",
            "tickets": [],
            "ticket_context": None,
            "rag_query": "",
            "business_status": None,
            "reject_info": None,
            "business_snapshot": None,
            "material_mapping": None,
            "retrieval_filter": {},
            "retrieved_sources": [],
            "rag_status": {},
            "evidence_ok": False,
            "evidence_decision": {},
            "refuse_reason": None,
            "answer": "",
            "refused": False,
            "suggested_actions": [],
            "events": [],
            "audit_payload": {},
        }

    def collect_sub_question(self, state: DeclarationQAState, question: dict[str, Any], index: int) -> dict[str, Any]:
        """收集单个子问题的路由、业务上下文和证据。"""
        state = self.intent_route(state)
        if state.get("route") == ROUTE_TICKET_CONTEXT_RELATED:
            state = self.ticket_list_query(state)
            if not state.get("refuse_reason"):
                state = self.ticket_selection(state)
            if not state.get("refuse_reason"):
                state = self.ticket_detail_query(state)
            if not state.get("refuse_reason"):
                state = self.ticket_context_build(state)
                state = self.ticket_knowledge_rag(state)
                state = self.evidence_check(state)
        elif state.get("route") == ROUTE_TICKET_CONTEXT_UNRELATED:
            state = self.knowledge_filter_build(state)
            state = self.hybrid_retrieve(state)
            state = self.evidence_check(state)
        if state.get("refuse_reason") and not state.get("answer"):
            state = self.refuse_or_human(state)
        status = "ready" if state.get("evidence_ok") and not state.get("refused") else "refused"
        if state.get("need_ticket_selection"):
            status = "need_ticket_selection"
        return {
            "id": question.get("id") or f"q{index}",
            "original_query": question.get("original_query") or question.get("query") or "",
            "query": state.get("query") or question.get("query") or "",
            "requires_ticket_context": bool(question.get("requires_ticket_context")),
            "is_expanded": bool(question.get("is_expanded")),
            "expansion_reason": question.get("expansion_reason") or "",
            "route": state.get("route") or "",
            "intent": state.get("intent") or "",
            "sub_intent": state.get("sub_intent") or "",
            "route_decision": state.get("route_decision") or {},
            "retrieval_chain": state.get("retrieval_chain") or "",
            "status": status,
            "need_ticket_selection": bool(state.get("need_ticket_selection")),
            "ticket_selection_reason": state.get("ticket_selection_reason") or "",
            "tickets": state.get("tickets") or [],
            "ticket_context": state.get("ticket_context") or {},
            "business_snapshot": state.get("business_snapshot") or {},
            "retrieval_filter": state.get("retrieval_filter") or {},
            "rag_query": state.get("rag_query") or state.get("normalized_query") or "",
            "sources": state.get("retrieved_sources") or [],
            "rag_status": state.get("rag_status") or {},
            "evidence_decision": state.get("evidence_decision") or {},
            "refused": bool(state.get("refused")),
            "refuse_reason": state.get("refuse_reason"),
            "answer": state.get("answer") or "",
        }

    def intent_route(self, state: DeclarationQAState) -> DeclarationQAState:
        """根据 question planner 输出映射路由并派生内部意图。"""
        next_state = dict(state)
        if next_state.get("refuse_reason"):
            next_state["route"] = ROUTE_REFUSE
            next_state["retrieval_chain"] = KNOWLEDGE_RETRIEVAL_CHAIN
            add_event(next_state, "intent", {"route": ROUTE_REFUSE, "reason": next_state.get("refuse_reason")})
            return next_state
        query = str(next_state.get("normalized_query") or next_state.get("query") or "")
        intent = classify_intent(query)
        if intent == "unsafe":
            route = ROUTE_REFUSE
            next_state["refuse_reason"] = "unsafe_or_out_of_scope"
        elif intent == "clarification":
            route = ROUTE_CLARIFICATION
            next_state["refuse_reason"] = "clarification_required"
        else:
            route = ROUTE_TICKET_CONTEXT_RELATED if bool(next_state.get("requires_ticket_context")) else ROUTE_TICKET_CONTEXT_UNRELATED
        route_decision = {
            "route": route,
            "intent": intent,
            "sub_intent": intent,
            "confidence": 1.0,
            "requires_ticket_context": route == ROUTE_TICKET_CONTEXT_RELATED,
            "can_answer_from_knowledge": route == ROUTE_TICKET_CONTEXT_UNRELATED or intent not in {INTENT_TICKET_STATUS, INTENT_REJECTION_REASON},
            "reason": "question_planner_requires_ticket_context",
            "risk_level": "block" if route == ROUTE_REFUSE else "pass",
            "entities": {"router_provider": "question_planner"},
        }
        next_state["route_decision"] = route_decision
        next_state["route"] = route
        next_state["intent"] = intent
        next_state["sub_intent"] = intent
        next_state["retrieval_chain"] = TICKET_AUGMENTED_CHAIN if route == ROUTE_TICKET_CONTEXT_RELATED else KNOWLEDGE_RETRIEVAL_CHAIN
        add_event(next_state, "intent", route_decision)
        return next_state

    def ticket_list_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询用户办件列表并写入状态。"""
        next_state = dict(state)
        if next_state.get("apply_context"):
            next_state["tickets"] = [ticket_summary_from_context(next_state["apply_context"] or {})]
            add_event(next_state, "ticket_list", {"count": 1, "source": "apply_context"})
            return next_state
        if not hasattr(self.business_adapter, "list_applications"):
            next_state["tickets"] = []
            next_state["refuse_reason"] = "ticket_not_found"
            add_event(next_state, "ticket_list", {"count": 0, "source": "adapter_missing"})
            return next_state
        tickets = self.business_adapter.list_applications(
            user_id=int(next_state["user_id"]),
            biz_type=next_state.get("biz_type"),
            region_code=next_state.get("region_code"),
        )
        if not tickets and next_state.get("biz_type"):
            tickets = self.business_adapter.list_applications(
                user_id=int(next_state["user_id"]),
                biz_type=None,
                region_code=next_state.get("region_code"),
            )
        next_state["tickets"] = tickets
        if not tickets:
            next_state["refuse_reason"] = "ticket_not_found"
        add_event(next_state, "ticket_list", {"count": len(tickets), "biz_type": next_state.get("biz_type")})
        return next_state

    def ticket_selection(self, state: DeclarationQAState) -> DeclarationQAState:
        """选择当前问题对应的办件。"""
        next_state = dict(state)
        if next_state.get("apply_context"):
            next_state["need_ticket_selection"] = False
            next_state["ticket_selection_reason"] = "provided_apply_context"
            add_event(next_state, "ticket_selected", {"reason": "provided_apply_context"})
            return next_state
        tickets = next_state.get("tickets") or []
        if len(tickets) == 1:
            ticket_id = str(tickets[0].get("ticket_id") or "")
            apply_context = self.business_adapter.get_apply_context(user_id=int(next_state["user_id"]), apply_id=ticket_id)
            if apply_context:
                next_state["apply_id"] = ticket_id
                next_state["apply_context"] = apply_context
                next_state["need_ticket_selection"] = False
                next_state["ticket_selection_reason"] = "single_candidate"
                add_event(next_state, "ticket_selected", {"ticket_id": ticket_id, "reason": "single_candidate"})
                return next_state
        if len(tickets) > 1:
            next_state["need_ticket_selection"] = True
            next_state["ticket_selection_reason"] = "multiple_candidates"
            next_state["refuse_reason"] = "ticket_selection_required"
            add_event(next_state, "ticket_selection_required", {"count": len(tickets)})
            return next_state
        next_state["need_ticket_selection"] = False
        next_state["ticket_selection_reason"] = "no_candidate"
        next_state["refuse_reason"] = "ticket_not_found"
        add_event(next_state, "ticket_selection_missing", {"reason": "no_candidate"})
        return next_state

    def ticket_detail_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询选中办件的详细上下文。"""
        next_state = dict(state)
        apply_context = next_state.get("apply_context") or {}
        if not apply_context:
            next_state["refuse_reason"] = "ticket_not_found"
            return next_state
        if next_state.get("intent") == INTENT_TICKET_STATUS:
            status = self.business_adapter.get_status(apply_context)
            if not (status.get("status") or status.get("current_node") or status.get("node_name")):
                next_state["business_status"] = {}
                next_state["business_snapshot"] = {}
                next_state["refuse_reason"] = "business_status_not_found"
                add_event(next_state, "business_status_missing", {"apply_id": apply_context.get("apply_id")})
                return next_state
            next_state["business_status"] = status
            next_state["business_snapshot"] = build_business_snapshot("status", status)
            add_event(next_state, "business_status", status)
            return next_state
        if next_state.get("intent") != INTENT_REJECTION_REASON:
            next_state["business_snapshot"] = build_business_snapshot("ticket_context", apply_context)
            add_event(next_state, "ticket_context_business", next_state["business_snapshot"])
            return next_state
        reject_info = self.business_adapter.get_reject_info(apply_context)
        if not (reject_info.get("reject_code") or reject_info.get("reject_reason") or reject_info.get("material_code")):
            next_state["reject_info"] = {}
            next_state["business_snapshot"] = {}
            next_state["refuse_reason"] = "business_reject_not_found"
            add_event(next_state, "business_reject_missing", {"apply_id": apply_context.get("apply_id")})
            return next_state
        next_state["reject_info"] = reject_info
        next_state["business_snapshot"] = build_business_snapshot("reject", reject_info)
        next_state["material_mapping"] = {
            "material_code": reject_info.get("material_code") or "",
            "material_name": reject_info.get("material_name") or "",
            "reject_code": reject_info.get("reject_code") or "",
            "audit_node": reject_info.get("audit_node") or "",
            "knowledge_tags": material_tags(reject_info),
        }
        add_event(next_state, "business_reject", reject_info)
        return next_state

    def ticket_context_build(self, state: DeclarationQAState) -> DeclarationQAState:
        """构造办件增强检索上下文。"""
        next_state = dict(state)
        ticket_context = {
            "ticket": ticket_summary_from_context(next_state.get("apply_context") or {}),
            "business_snapshot": next_state.get("business_snapshot") or {},
            "material_mapping": next_state.get("material_mapping") or {},
        }
        next_state["ticket_context"] = ticket_context
        add_event(next_state, "ticket_context", ticket_context)
        return next_state

    def ticket_knowledge_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        """围绕办件问题执行知识检索。"""
        next_state = dict(state)
        filters = base_policy_filter(next_state, allow_request_filter=False)
        query = ticket_rag_query(next_state)
        result = self.rag_engine.retrieve(query, filters, top_k=4)
        next_state["rag_query"] = query
        next_state["retrieval_filter"] = filters
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        if next_state.get("intent") == INTENT_TICKET_STATUS:
            next_state["evidence_ok"] = True
            next_state["evidence_decision"] = {
                "ok": True,
                "reason": "business_status_verified",
                "business_snapshot_present": bool(next_state.get("business_snapshot")),
                "source_count": len(result.sources),
            }
        add_event(next_state, "rag_status", {**result.status, "rag_query": query})
        return next_state

    def status_apply_context_check(self, state: DeclarationQAState) -> DeclarationQAState:
        """检查状态查询是否具备办件上下文。"""
        next_state = dict(state)
        apply_context = next_state.get("apply_context") or {}
        if not apply_context or not str(apply_context.get("apply_id") or "").strip():
            next_state["refuse_reason"] = "apply_context_required_for_status"
        return next_state

    def reject_apply_context_check(self, state: DeclarationQAState) -> DeclarationQAState:
        """检查退回查询是否具备办件上下文。"""
        next_state = dict(state)
        apply_context = next_state.get("apply_context") or {}
        if not apply_context or not str(apply_context.get("apply_id") or "").strip():
            next_state["refuse_reason"] = "apply_context_required_for_reject"
        return next_state

    def business_status_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询办件状态业务数据。"""
        next_state = dict(state)
        status = self.business_adapter.get_status(next_state.get("apply_context") or {})
        if not (status.get("status") or status.get("current_node") or status.get("node_name")):
            next_state["business_status"] = {}
            next_state["business_snapshot"] = {}
            next_state["refuse_reason"] = "business_status_not_found"
            add_event(next_state, "business_status_missing", {"apply_id": (next_state.get("apply_context") or {}).get("apply_id")})
            return next_state
        next_state["business_status"] = status
        next_state["business_snapshot"] = build_business_snapshot("status", status)
        next_state["biz_type"] = next_state.get("biz_type") or status.get("biz_type")
        next_state["region_code"] = status.get("region_code") or next_state.get("region_code")
        add_event(next_state, "business_status", status)
        return next_state

    def status_explain_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        """检索办件状态解释知识。"""
        next_state = dict(state)
        status = next_state.get("business_status") or {}
        if not status:
            next_state["refused"] = True
            next_state["refuse_reason"] = "business_status_not_found"
            next_state["answer"] = "暂未从办件系统查询到该申请单的实时进度，请确认申请单是否有效，或联系人工客服核验。"
            add_event(next_state, "refused", {"reason": next_state["refuse_reason"]})
            return next_state
        filters = base_policy_filter(next_state, allow_request_filter=False)
        next_state["retrieval_filter"] = filters
        query = f"{status.get('node_name') or status.get('current_node') or ''} {next_state.get('normalized_query') or ''}"
        result = self.rag_engine.retrieve(query, filters, top_k=3)
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        next_state["evidence_ok"] = True
        next_state["evidence_decision"] = {
            "ok": True,
            "reason": "business_status_verified",
            "business_snapshot_present": bool(status),
            "source_count": len(result.sources),
        }
        add_event(next_state, "rag_status", result.status)
        return next_state

    def business_reject_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询办件退回补正业务数据。"""
        next_state = dict(state)
        reject_info = self.business_adapter.get_reject_info(next_state.get("apply_context") or {})
        if not (reject_info.get("reject_code") or reject_info.get("reject_reason") or reject_info.get("material_code")):
            next_state["reject_info"] = {}
            next_state["business_snapshot"] = {}
            next_state["refuse_reason"] = "business_reject_not_found"
            add_event(next_state, "business_reject_missing", {"apply_id": (next_state.get("apply_context") or {}).get("apply_id")})
            return next_state
        next_state["reject_info"] = reject_info
        next_state["business_snapshot"] = build_business_snapshot("reject", reject_info)
        next_state["biz_type"] = next_state.get("biz_type") or reject_info.get("biz_type")
        next_state["region_code"] = reject_info.get("region_code") or next_state.get("region_code")
        add_event(next_state, "business_reject", reject_info)
        return next_state

    def material_mapping(self, state: DeclarationQAState) -> DeclarationQAState:
        """从退回信息中生成材料检索条件。"""
        next_state = dict(state)
        reject = next_state.get("reject_info") or {}
        if not reject:
            next_state["refuse_reason"] = "business_reject_not_found"
            add_event(next_state, "material_mapping_skipped", {"reason": "business_reject_not_found"})
            return next_state
        mapping = {
            "material_code": reject.get("material_code") or "",
            "material_name": reject.get("material_name") or "",
            "reject_code": reject.get("reject_code") or "",
            "audit_node": reject.get("audit_node") or "",
            "knowledge_tags": material_tags(reject),
        }
        next_state["material_mapping"] = mapping
        add_event(next_state, "material_mapping", mapping)
        return next_state

    def material_rule_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        """检索补正材料规则知识。"""
        next_state = dict(state)
        reject = next_state.get("reject_info") or {}
        if not reject:
            next_state["retrieval_filter"] = {}
            next_state["retrieved_sources"] = []
            next_state["rag_status"] = {"skipped": True, "reason": "business_reject_not_found"}
            add_event(next_state, "rag_status", next_state["rag_status"])
            return next_state
        filters = base_policy_filter(next_state, allow_request_filter=False)
        query = " ".join(
            str(item)
            for item in [
                reject.get("reject_reason"),
                reject.get("material_name"),
                reject.get("reject_code"),
                next_state.get("normalized_query"),
            ]
            if item
        )
        result = self.rag_engine.retrieve(query, filters, top_k=4)
        next_state["retrieval_filter"] = filters
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", result.status)
        return next_state

    def knowledge_filter_build(self, state: DeclarationQAState) -> DeclarationQAState:
        """构造普通知识检索过滤条件。"""
        next_state = dict(state)
        if not next_state.get("biz_type"):
            next_state["biz_type"] = infer_biz_type(next_state.get("normalized_query") or "")
        filters = base_policy_filter(next_state, allow_request_filter=True)
        next_state["retrieval_filter"] = filters
        add_event(next_state, "knowledge_filter", filters)
        return next_state

    def hybrid_retrieve(self, state: DeclarationQAState) -> DeclarationQAState:
        """执行普通知识问答混合检索。"""
        next_state = dict(state)
        query = knowledge_rag_query(next_state)
        result = self.rag_engine.retrieve(
            query,
            next_state.get("retrieval_filter") or {},
            top_k=4,
        )
        next_state["rag_query"] = query
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", {**result.status, "rag_query": query})
        return next_state

    def evidence_check(self, state: DeclarationQAState) -> DeclarationQAState:
        """执行证据门控检查。"""
        next_state = dict(state)
        if next_state.get("route") == ROUTE_TICKET_CONTEXT_RELATED and next_state.get("evidence_ok"):
            add_event(next_state, "evidence", next_state.get("evidence_decision") or {})
            return next_state
        decision = decide_evidence(
            next_state.get("retrieved_sources") or [],
            next_state.get("retrieval_filter") or {},
            intent=str(next_state.get("intent") or ""),
            chain=str(next_state.get("retrieval_chain") or KNOWLEDGE_RETRIEVAL_CHAIN),
            business_payload=next_state.get("business_snapshot") or {},
        )
        next_state["evidence_ok"] = decision.ok
        next_state["evidence_decision"] = decision.payload()
        if not decision.ok:
            next_state["refuse_reason"] = decision.reason
        add_event(next_state, "evidence", decision.payload())
        return next_state

    def refuse_or_human(self, state: DeclarationQAState) -> DeclarationQAState:
        """在证据不足时生成拒答和人工建议。"""
        next_state = dict(state)
        next_state["refused"] = True
        reason = next_state.get("refuse_reason") or "out_of_scope"
        next_state["refuse_reason"] = reason
        if reason == "apply_context_required_for_status":
            answer = "该问题需要绑定具体申请单。请先从办件列表选择对应申请单后再查询进度。"
        elif reason == "apply_context_required_for_reject":
            answer = "该问题需要绑定具体申请单。请先选择被退回的申请单后再查看驳回原因。"
        elif reason == "business_status_not_found":
            answer = "暂未从办件系统查询到该申请单的实时进度，请确认申请单是否有效，或联系人工客服核验。"
        elif reason == "business_reject_not_found":
            answer = "暂未从办件系统查询到该申请单的有效驳回或补正记录，不能根据模型推测驳回原因。建议查看申请单详情或联系人工客服。"
        elif reason == "ticket_selection_required":
            answer = "我查到多个可能相关的申请单，请先选择要咨询的工单后再继续查询。"
        elif reason == "ticket_not_found":
            answer = "暂未查询到当前用户下可匹配的申请单。你可以切换到通用知识咨询，或联系人工客服核验。"
        elif reason == "clarification_required":
            answer = "请补充你想咨询的事项名称、地区或具体问题，例如材料清单、办理流程、办理进度或驳回原因。"
        elif reason == "unsafe_or_out_of_scope":
            answer = "该问题不属于申报辅助范围，或涉及不合规操作，暂不能提供处理建议。"
        elif reason == "apply_access_denied":
            answer = "当前登录用户无权查看该申请单，请确认申请单是否属于本人或重新登录后再试。"
        else:
            answer = "暂未查询到与当前地区和业务类型完全匹配的办理依据，建议查看办事指南或联系人工客服确认。"
        next_state["answer"] = answer
        add_event(next_state, "refused", {"reason": reason})
        return next_state

    def llm_generate(self, state: DeclarationQAState) -> DeclarationQAState:
        """调用回答生成器生成最终回答。"""
        next_state = dict(state)
        answer, prompt = self.answer_generator.generate(next_state)
        next_state["answer"] = answer
        audit_payload = dict(next_state.get("audit_payload") or {})
        audit_payload["final_prompt"] = prompt
        next_state["audit_payload"] = audit_payload
        add_event(next_state, "answer_generated", {"intent": next_state.get("intent")})
        return next_state

    def answer_postprocess(self, state: DeclarationQAState) -> DeclarationQAState:
        """补充默认回答并整理输出。"""
        next_state = dict(state)
        next_state["suggested_actions"] = suggested_actions(next_state)
        add_event(next_state, "suggested_actions", {"items": next_state["suggested_actions"]})
        return next_state

    def audit_log(self, state: DeclarationQAState) -> DeclarationQAState:
        """写入问答审计日志。"""
        next_state = dict(state)
        payload = {
            "user_id": next_state.get("user_id"),
            "session_id": next_state.get("session_id"),
            "query": next_state.get("query"),
            "normalized_query": next_state.get("normalized_query"),
            "intent": next_state.get("intent"),
            "sub_intent": next_state.get("sub_intent"),
            "route": next_state.get("route"),
            "route_decision": next_state.get("route_decision") or {},
            "question_plan": next_state.get("question_plan") or {},
            "multi_question": next_state.get("multi_question"),
            "sub_tasks": next_state.get("sub_tasks") or [],
            "retrieval_chain": next_state.get("retrieval_chain"),
            "region_code": next_state.get("region_code"),
            "biz_type": next_state.get("biz_type"),
            "apply_id": (next_state.get("apply_context") or {}).get("apply_id") if next_state.get("apply_context") else None,
            "need_ticket_selection": next_state.get("need_ticket_selection"),
            "ticket_selection_reason": next_state.get("ticket_selection_reason"),
            "tickets": next_state.get("tickets") or [],
            "ticket_context": next_state.get("ticket_context") or {},
            "business_snapshot": next_state.get("business_snapshot"),
            "business_status": next_state.get("business_status"),
            "reject_info": next_state.get("reject_info"),
            "material_mapping": next_state.get("material_mapping"),
            "retrieval_filter": next_state.get("retrieval_filter"),
            "rag_query": next_state.get("rag_query") or "",
            "retrieved_sources": next_state.get("retrieved_sources") or [],
            "rag_status": next_state.get("rag_status"),
            "evidence_ok": next_state.get("evidence_ok"),
            "evidence_decision": next_state.get("evidence_decision") or {},
            "refused": next_state.get("refused"),
            "refuse_reason": next_state.get("refuse_reason"),
            "answer": next_state.get("answer"),
            "suggested_actions": next_state.get("suggested_actions") or [],
            "events": next_state.get("events") or [],
            **(next_state.get("audit_payload") or {}),
        }
        audit_id = self.audit_store.append(payload)
        next_state["audit_id"] = audit_id
        add_event(next_state, "audit_logged", {"audit_id": audit_id})
        return next_state


def ticket_summary_from_context(context: dict[str, Any]) -> dict[str, Any]:
    """从办件上下文生成摘要。"""
    return {
        "ticket_id": str(context.get("apply_id") or ""),
        "biz_type": str(context.get("biz_type") or ""),
        "region_code": str(context.get("region_code") or ""),
        "status": str(context.get("status") or ""),
        "current_node": str(context.get("current_node") or ""),
        "node_name": str(context.get("node_name") or ""),
        "title": business_title(context),
        "updated_at": str(context.get("updated_at") or ""),
    }


def ticket_title_from_biz_type(biz_type: str) -> str:
    """根据业务类型生成办件名称。"""
    names = {
        "health_certificate": "健康证办理",
        "residence_permit": "居住证办理",
        "id_card_replace": "身份证换领",
    }
    return names.get(str(biz_type or ""), str(biz_type or "政务服务事项"))


def ticket_rag_query(state: DeclarationQAState) -> str:
    """构造办件增强检索查询文本。"""
    business = state.get("business_status") or state.get("reject_info") or {}
    apply_context = state.get("apply_context") or {}
    mapping = state.get("material_mapping") or {}
    parts = [
        business_title(business, apply_context, state),
        business.get("node_name"),
        business.get("current_node"),
        business.get("reject_reason"),
        business.get("material_name"),
        business.get("reject_code"),
        mapping.get("material_code"),
        state.get("normalized_query"),
    ]
    return normalize_text(" ".join(str(item) for item in parts if item))


def knowledge_rag_query(state: DeclarationQAState) -> str:
    """构造普通知识检索查询，已选办件时继承事项范围。"""
    query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
    scope = selected_ticket_scope_text(state)
    if not scope:
        return query
    return normalize_text(f"{scope} {query}")


def selected_ticket_scope_text(state: DeclarationQAState) -> str:
    """从当前办件提取事项范围，不包含节点、退回原因等工单事实。"""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return ""
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    parts = [
        business_title(apply_context, state),
        payload.get("official_item_name"),
        payload.get("item_name"),
        payload.get("title"),
        payload.get("catalog_name"),
        payload.get("source_file"),
        payload.get("item_code"),
        apply_context.get("biz_type"),
    ]
    return normalize_text(" ".join(unique_strings([str(item) for item in parts if item])))


def business_title(*sources: dict[str, Any]) -> str:
    """生成业务类型展示名称。"""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("title", "item_name", "itemName", "catalog_name", "catalogName"):
            title = str(source.get(key) or "").strip()
            if title:
                return title
        payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
        for key in ("item_name", "title", "itemName", "catalog_name", "catalogName"):
            title = str(payload.get(key) or "").strip()
            if title:
                return title
    for source in sources:
        if isinstance(source, dict):
            title = ticket_title_from_biz_type(str(source.get("biz_type") or ""))
            if title:
                return title
    return "政务服务事项"


def base_policy_filter(state: DeclarationQAState, *, allow_request_filter: bool = True) -> dict[str, Any]:
    """构造基础政策检索过滤条件。"""
    return {}


def build_business_snapshot(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """构造可用于回答的办件快照。"""
    cleaned = {key: value for key, value in (payload or {}).items() if value not in ("", None, [], {})}
    if not cleaned:
        return {}
    return {
        "type": kind,
        "source": "business_adapter",
        "apply_id": cleaned.get("apply_id"),
        "biz_type": cleaned.get("biz_type"),
        "title": business_title(cleaned),
        "region_code": cleaned.get("region_code"),
        "status": cleaned.get("status"),
        "current_node": cleaned.get("current_node"),
        "node_name": cleaned.get("node_name"),
        "material_code": cleaned.get("material_code"),
        "material_name": cleaned.get("material_name"),
        "reject_code": cleaned.get("reject_code"),
        "reject_reason": cleaned.get("reject_reason"),
        "audit_node": cleaned.get("audit_node"),
        "submitted_at": cleaned.get("submitted_at"),
        "completed_at": cleaned.get("completed_at"),
        "updated_at": cleaned.get("updated_at"),
        "payload": redact_business_payload(cleaned.get("payload") or {}),
    }


def redact_business_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """脱敏业务载荷中的敏感字段。"""
    redacted = {}
    for key, value in (payload or {}).items():
        lowered = str(key).lower()
        if any(token in lowered for token in ["id_card", "idcard", "phone", "mobile", "name", "address"]):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted


def add_event(state: dict[str, Any], event: str, data: dict[str, Any]) -> None:
    """向工作流事件列表追加事件。"""
    events = list(state.get("events") or [])
    events.append({"event": event, "data": data})
    state["events"] = events


def apply_planned_question(state: dict[str, Any], question: dict[str, Any]) -> None:
    """将 planner 理解后的单问题写回状态，保留原始问题在 question_plan 中。"""
    planned_query = normalize_text(str(question.get("query") or ""))
    if not planned_query:
        return
    state["query"] = planned_query
    state["normalized_query"] = planned_query
    state["requires_ticket_context"] = bool(question.get("requires_ticket_context"))
    state["biz_type"] = state.get("biz_type") or infer_biz_type(planned_query)


def unique_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 chunk/doc 去重来源。"""
    seen = set()
    output = []
    for source in sources:
        key = (source.get("chunk_id") or "", source.get("doc_id") or "", source.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        output.append(source)
    return output


def unique_tickets(tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 ticket_id 去重办件摘要。"""
    seen = set()
    output = []
    for ticket in tickets:
        ticket_id = str(ticket.get("ticket_id") or "")
        if not ticket_id or ticket_id in seen:
            continue
        seen.add(ticket_id)
        output.append(ticket)
    return output


def unique_strings(values: list[str]) -> list[str]:
    """按出现顺序去重字符串。"""
    seen = set()
    output = []
    for value in values:
        item = str(value or "")
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


STREAM_TOKEN_SIZE = 3
STREAM_TOKEN_DELAY_SECONDS = 0.02


def token_events(
    text: str,
    *,
    size: int = STREAM_TOKEN_SIZE,
    delay_seconds: float = STREAM_TOKEN_DELAY_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> Iterable[dict[str, Any]]:
    """将文本切分为 token 事件。"""
    if sleep is None:
        from time import sleep as sleep_func

        sleep = sleep_func
    value = str(text or "")
    for index in range(0, len(value), size):
        if index > 0 and delay_seconds > 0:
            sleep(delay_seconds)
        yield {"event": "token", "data": {"content": value[index : index + size]}}


def mask_sensitive_text(value: str) -> str:
    """遮盖文本中的敏感证件和手机号信息。"""
    text = str(value or "")
    text = re.sub(r"\b\d{17}[\dXx]\b", "[id_card]", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "[phone]", text)
    return text


def classify_intent(query: str) -> str:
    """按查询文本粗分类业务意图。"""
    text = str(query or "")
    if any(keyword in text for keyword in ["伪造", "造假", "假材料", "绕过审核"]):
        return "unsafe"
    if len(text.strip()) < 3:
        return "clarification"
    if any(keyword in text for keyword in ["退回", "驳回", "补正", "不通过", "被拒", "被退"]):
        return INTENT_REJECTION_REASON
    if any(keyword in text for keyword in ["进度", "到哪", "状态", "审核到", "办到哪", "多久能办完", "什么情况", "怎么样"]):
        return INTENT_TICKET_STATUS
    if any(keyword in text for keyword in ["材料", "要带", "提交什么", "需要什么", "清单"]):
        return "material_requirements"
    if any(keyword in text for keyword in ["流程", "步骤", "怎么办", "如何办理", "怎么申请"]):
        return "process_steps"
    if any(keyword in text for keyword in ["条件", "资格", "能不能办", "是否可以"]):
        return "eligibility"
    if any(keyword in text for keyword in ["费用", "收费", "多少钱", "时限", "多久", "几天", "多少天", "拿结果", "办结"]):
        return "fee_time_limit"
    if any(keyword in text for keyword in ["地点", "窗口", "地址", "哪里办"]):
        return "office_location"
    if any(keyword in text for keyword in ["电话", "咨询", "投诉"]):
        return "contact"
    if any(keyword in text for keyword in ["网上办", "入口", "链接", "在线"]):
        return "online_apply"
    if any(keyword in text for keyword in ["依据", "政策", "法规", "文件"]):
        return "policy_basis"
    if any(keyword in text for keyword in ["表格", "模板", "样表", "申请表"]):
        return "form_template"
    return "knowledge_qa"


def infer_biz_type(query: str) -> str | None:
    """根据文本推断业务类型。"""
    text = str(query or "")
    if "健康证" in text:
        return "health_certificate"
    if "居住证" in text or "居住证明" in text:
        return "residence_permit"
    if "身份证" in text:
        return "id_card_replace"
    return None


def material_tags(reject: dict[str, Any]) -> list[str]:
    """根据退回信息生成材料标签。"""
    material_code = str(reject.get("material_code") or "")
    reject_code = str(reject.get("reject_code") or "")
    tags = []
    if "PHOTO" in reject_code or "PHOTO" in material_code:
        tags.extend(["照片规范", "图片清晰度", "上传规范"])
    if material_code:
        tags.append(material_code)
    if reject_code:
        tags.append(reject_code)
    return tags


def suggested_actions(state: DeclarationQAState) -> list[dict[str, Any]]:
    """根据状态生成后续建议动作。"""
    app_version = state.get("app_version") or ""
    apply_context = state.get("apply_context") or {}
    if state.get("refused"):
        return [{"type": "human_service", "label": "联系人工客服", "payload": {"appVersion": app_version}}]
    if state.get("intent") in {INTENT_TICKET_STATUS}:
        return [
            {
                "type": "open_apply_detail",
                "label": "查看申请单详情",
                "payload": {"applyId": apply_context.get("apply_id"), "appVersion": app_version},
            }
        ]
    if state.get("intent") in {INTENT_REJECTION_REASON}:
        reject_info = state.get("reject_info") or {}
        return [
            {
                "type": "open_material_reupload",
                "label": "重新上传补正材料",
                "payload": {
                    "applyId": apply_context.get("apply_id"),
                    "materialCode": reject_info.get("material_code"),
                    "appVersion": app_version,
                },
            }
        ]
    return [
        {
            "type": "open_guide",
            "label": "查看办事指南",
            "payload": {"bizType": state.get("biz_type"), "appVersion": app_version},
        }
    ]
