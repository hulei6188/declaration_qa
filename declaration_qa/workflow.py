"""申报问答 LangGraph 工作流和业务节点实现。"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from uuid import uuid4

from langgraph.config import get_config, get_stream_writer
from langgraph.graph import END, START, StateGraph

from collections.abc import Callable, Iterable

from declaration_qa.audit import InMemoryAuditStore
from declaration_qa.business import BusinessDataAdapter, InMemoryBusinessDataAdapter
from declaration_qa.domain import RetrievalResult, RetrievedSource
from declaration_qa.evidence import decide_evidence
from declaration_qa.prompting import DeterministicAnswerGenerator
from declaration_qa.question_planner import RuleQuestionPlanner
from declaration_qa.query_rewriter import RuleQueryRewriter
from declaration_qa.rag_engine import InMemoryRAGEngine, build_seed_rag_engine, normalize_text
from declaration_qa.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, TICKET_AUGMENTED_CHAIN, normalize_retrieval_filter
from declaration_qa.router import (
    INTENT_CORRECTION_GUIDANCE,
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
        query_rewriter: RuleQueryRewriter | None = None,
        max_sub_question_workers: int = 4,
        max_retrieval_workers: int = 4,
    ) -> None:
        """初始化申报问答工作流。"""
        self.rag_engine = rag_engine or build_seed_rag_engine()
        self.business_adapter = business_adapter or InMemoryBusinessDataAdapter()
        self.audit_store = audit_store or InMemoryAuditStore()
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.question_planner = question_planner or RuleQuestionPlanner()
        self.query_rewriter = query_rewriter or RuleQueryRewriter()
        self.max_sub_question_workers = max(1, int(max_sub_question_workers))
        self.max_retrieval_workers = max(1, int(max_retrieval_workers))
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
        """通过 LangGraph 流式执行问答工作流并产出 SSE 事件。"""
        state: DeclarationQAState = self._initial_state(request)
        yield {"event": "run_started", "data": {"session_id": state["session_id"]}}

        emitted_event_count = 0
        streamed_answer = False
        config = {"configurable": {"declaration_qa_stream": True}}
        for chunk in self.graph.stream(state, config=config, stream_mode=["updates", "custom"]):
            mode, payload = graph_stream_part(chunk)
            if mode == "custom":
                event = normalize_stream_event(payload)
                if event:
                    if event["event"] == "token":
                        streamed_answer = True
                    yield event
                continue
            if mode != "updates" or not isinstance(payload, dict):
                continue
            for node_name, node_state in payload.items():
                if not isinstance(node_state, dict):
                    continue
                state.update(node_state)
                events = state.get("events") or []
                for event in events[emitted_event_count:]:
                    yield {"event": event.get("event") or node_name, "data": event.get("data") or {}}
                emitted_event_count = len(events)
                if should_emit_sources(node_name, state):
                    yield {"event": "sources", "data": {"items": state.get("retrieved_sources") or []}}
                if node_name == "refuse_or_human":
                    yield from token_events(str(state.get("answer") or ""))
                elif node_name == "llm_generate" and not streamed_answer:
                    yield from token_events(str(state.get("answer") or ""))

        yield done_event(state)

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
            "request_biz_type": request.biz_type,
            "apply_id": request.apply_id,
            "apply_context": apply_context,
            "metadata_filter": request.metadata_filter,
            "intent": "unknown",
            "sub_intent": "",
            "route": "",
            "route_decision": {},
            "question_plan": {},
            "requires_ticket_context": False,
            "ticket_fact_type": "none",
            "retrieval_focus": "knowledge_qa",
            "multi_question": False,
            "sub_tasks": [],
            "retrieval_chain": "",
            "need_ticket_selection": False,
            "ticket_selection_reason": "",
            "tickets": [],
            "ticket_context": None,
            "retrieval_query": "",
            "rag_query": "",
            "query_rewrite": {},
            "retrieval_queries": [],
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
        builder.add_node("prepare_task", self.prepare_task)
        builder.add_node("ticket_list_query", self.ticket_list_query)
        builder.add_node("ticket_selection", self.ticket_selection)
        builder.add_node("ticket_detail_query", self.ticket_detail_query)
        builder.add_node("ticket_context_build", self.ticket_context_build)
        builder.add_node("query_rewrite", self.query_rewrite)
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
            {"multi": "multi_question_collect", "single": "prepare_task"},
        )
        builder.add_conditional_edges(
            "multi_question_collect",
            lambda state: "needs_selection" if state.get("need_ticket_selection") else "ready",
            {"needs_selection": "refuse_or_human", "ready": "llm_generate"},
        )
        builder.add_conditional_edges(
            "prepare_task",
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
        builder.add_edge("ticket_context_build", "knowledge_filter_build")
        builder.add_edge("knowledge_filter_build", "query_rewrite")
        builder.add_conditional_edges(
            "query_rewrite",
            lambda state: "ticket" if state.get("route") == ROUTE_TICKET_CONTEXT_RELATED else "knowledge",
            {"ticket": "ticket_knowledge_rag", "knowledge": "hybrid_retrieve"},
        )
        builder.add_edge("ticket_knowledge_rag", "evidence_check")
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
            question["retrieval_query"] = normalize_text(
                str(question.get("retrieval_query") or question.get("query") or question.get("original_query") or "")
            )
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
        planned_questions = list(next_state.get("sub_tasks") or [])
        if len(planned_questions) > 1 and self.max_sub_question_workers > 1:
            task_results: list[dict[str, Any] | None] = [None] * len(planned_questions)
            with ThreadPoolExecutor(max_workers=min(self.max_sub_question_workers, len(planned_questions))) as executor:
                futures = {
                    executor.submit(
                        self.collect_sub_question,
                        self.state_for_sub_question(next_state, question),
                        question,
                        index,
                    ): index - 1
                    for index, question in enumerate(planned_questions, start=1)
                }
                for future in as_completed(futures):
                    task_results[futures[future]] = future.result()
            sub_tasks = [task for task in task_results if task is not None]
            collection_mode = "parallel"
        else:
            collection_mode = "serial"
            for index, question in enumerate(planned_questions, start=1):
                task_state = self.state_for_sub_question(next_state, question)
                task = self.collect_sub_question(task_state, question, index)
                sub_tasks.append(task)

        for task in sub_tasks:
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
                "collection_mode": collection_mode,
                "sub_question_parallelism": min(self.max_sub_question_workers, max(1, len(planned_questions))),
            },
        )
        return next_state

    def state_for_sub_question(self, state: DeclarationQAState, question: dict[str, Any]) -> DeclarationQAState:
        """构造子问题状态，复用原始请求上下文。"""
        query = normalize_text(str(question.get("original_query") or question.get("query") or ""))
        retrieval_query = normalize_text(str(question.get("retrieval_query") or query))
        inherited_apply_context = state.get("apply_context")
        inherited_biz_type = (
            (inherited_apply_context or {}).get("biz_type")
            if inherited_apply_context
            else infer_biz_type(retrieval_query) or infer_biz_type(query) or state.get("request_biz_type")
        )
        return {
            **state,
            "query": query,
            "normalized_query": retrieval_query,
            "retrieval_query": retrieval_query,
            "requires_ticket_context": bool(question.get("requires_ticket_context")),
            "biz_type": inherited_biz_type,
            "multi_question": False,
            "sub_tasks": [],
            "question_plan": {},
            "intent": "unknown",
            "sub_intent": "",
            "route": "",
            "route_decision": {},
            "ticket_fact_type": question.get("ticket_fact_type") or "none",
            "retrieval_focus": question.get("retrieval_focus") or "knowledge_qa",
            "retrieval_chain": "",
            "need_ticket_selection": False,
            "ticket_selection_reason": "",
            "tickets": [],
            "ticket_context": None,
            "retrieval_query": retrieval_query,
            "rag_query": "",
            "query_rewrite": {},
            "retrieval_queries": [],
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
        state = self.prepare_task(state)
        if state.get("route") == ROUTE_TICKET_CONTEXT_RELATED:
            state = self.ticket_list_query(state)
            if not state.get("refuse_reason"):
                state = self.ticket_selection(state)
            if not state.get("refuse_reason"):
                state = self.ticket_detail_query(state)
            if not state.get("refuse_reason"):
                state = self.ticket_context_build(state)
                state = self.knowledge_filter_build(state)
                state = self.query_rewrite(state)
                state = self.ticket_knowledge_rag(state)
                state = self.evidence_check(state)
        elif state.get("route") == ROUTE_TICKET_CONTEXT_UNRELATED:
            state = self.knowledge_filter_build(state)
            state = self.query_rewrite(state)
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
            "retrieval_query": state.get("retrieval_query") or question.get("retrieval_query") or "",
            "requires_ticket_context": bool(question.get("requires_ticket_context")),
            "ticket_fact_type": state.get("ticket_fact_type") or question.get("ticket_fact_type") or "none",
            "retrieval_focus": state.get("retrieval_focus") or question.get("retrieval_focus") or "knowledge_qa",
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
            "query_rewrite": state.get("query_rewrite") or {},
            "retrieval_queries": state.get("retrieval_queries") or [],
            "sources": state.get("retrieved_sources") or [],
            "rag_status": state.get("rag_status") or {},
            "evidence_decision": state.get("evidence_decision") or {},
            "refused": bool(state.get("refused")),
            "refuse_reason": state.get("refuse_reason"),
            "answer": state.get("answer") or "",
        }

    def prepare_task(self, state: DeclarationQAState) -> DeclarationQAState:
        """根据 Planner 任务字段准备兼容展示字段和链路开关。"""
        next_state = dict(state)
        if next_state.get("refuse_reason"):
            next_state["route"] = ROUTE_REFUSE
            next_state["retrieval_chain"] = KNOWLEDGE_RETRIEVAL_CHAIN
            add_event(next_state, "task_prepared", {"route": ROUTE_REFUSE, "reason": next_state.get("refuse_reason")})
            return next_state
        query = str(next_state.get("normalized_query") or next_state.get("query") or "")
        retrieval_focus = str(next_state.get("retrieval_focus") or "").strip() or classify_intent(query)
        ticket_fact_type = str(next_state.get("ticket_fact_type") or "none").strip() or "none"
        if is_unsafe_query(query):
            route = ROUTE_REFUSE
            intent = "unsafe"
            next_state["refuse_reason"] = "unsafe_or_out_of_scope"
        elif len(query.strip()) < 3:
            route = ROUTE_CLARIFICATION
            intent = "clarification"
            next_state["refuse_reason"] = "clarification_required"
        else:
            intent = task_intent(retrieval_focus, ticket_fact_type, bool(next_state.get("requires_ticket_context")), query)
            route = ROUTE_TICKET_CONTEXT_RELATED if bool(next_state.get("requires_ticket_context")) else ROUTE_TICKET_CONTEXT_UNRELATED
        route_decision = {
            "route": route,
            "intent": intent,
            "sub_intent": intent,
            "confidence": 1.0,
            "requires_ticket_context": route == ROUTE_TICKET_CONTEXT_RELATED,
            "ticket_fact_type": ticket_fact_type,
            "retrieval_focus": retrieval_focus,
            "can_answer_from_knowledge": route == ROUTE_TICKET_CONTEXT_UNRELATED or intent not in {INTENT_TICKET_STATUS, INTENT_REJECTION_REASON},
            "reason": "question_planner_task_fields",
            "risk_level": "block" if route == ROUTE_REFUSE else "pass",
            "entities": {"task_provider": "question_planner", "router_provider": "question_planner"},
        }
        next_state["route_decision"] = route_decision
        next_state["route"] = route
        next_state["intent"] = intent
        next_state["sub_intent"] = intent
        next_state["retrieval_focus"] = retrieval_focus
        next_state["ticket_fact_type"] = ticket_fact_type
        next_state["retrieval_chain"] = TICKET_AUGMENTED_CHAIN if route == ROUTE_TICKET_CONTEXT_RELATED else KNOWLEDGE_RETRIEVAL_CHAIN
        add_event(next_state, "intent", route_decision)
        add_event(next_state, "task_prepared", route_decision)
        return next_state

    def intent_route(self, state: DeclarationQAState) -> DeclarationQAState:
        """兼容旧测试和外部调用，核心链路不再使用该节点。"""
        return self.prepare_task(state)

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
        result = self.multi_retrieve(next_state, filters)
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
        add_event(next_state, "rag_status", {**result.status, "rag_query": next_state.get("rag_query") or ""})
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
        if not next_state.get("biz_type") and not next_state.get("apply_context"):
            next_state["biz_type"] = infer_biz_type(next_state.get("normalized_query") or "")
        filters = base_policy_filter(next_state, allow_request_filter=True)
        next_state["retrieval_filter"] = filters
        add_event(next_state, "knowledge_filter", filters)
        return next_state

    def query_rewrite(self, state: DeclarationQAState) -> DeclarationQAState:
        """生成多路检索 query。"""
        next_state = dict(state)
        rewrite = self.query_rewriter.rewrite(next_state)
        payload = rewrite.payload()
        next_state["query_rewrite"] = payload
        next_state["retrieval_queries"] = payload.get("queries") or []
        next_state["rag_query"] = "\n".join(str(query.get("text") or "") for query in next_state["retrieval_queries"])
        add_event(
            next_state,
            "query_rewrite",
            {
                "count": len(next_state["retrieval_queries"]),
                "retrieval_focus": payload.get("retrieval_focus"),
                "ticket_fact_type": payload.get("ticket_fact_type"),
                "provider": payload.get("provider"),
                "reason": payload.get("reason"),
                "scope_text": payload.get("scope_text") or "",
                "queries": next_state["retrieval_queries"],
            },
        )
        return next_state

    def multi_retrieve(self, state: DeclarationQAState, filters: dict[str, Any]) -> RetrievalResult:
        """按 rewrite 结果执行多路检索并去重合并。"""
        retrieval_queries = state.get("retrieval_queries") or []
        if not retrieval_queries:
            fallback_query = ticket_rag_query(state) if state.get("route") == ROUTE_TICKET_CONTEXT_RELATED else knowledge_rag_query(state)
            retrieval_queries = [{"id": "rq1", "text": fallback_query, "strategy": "fallback", "top_k": 4}]
        executable_queries = dedupe_retrieval_queries(retrieval_queries)
        query_statuses: list[dict[str, Any] | None] = [None] * len(executable_queries)

        def retrieve_one(retrieval_query: dict[str, Any]) -> tuple[dict[str, Any], RetrievalResult]:
            text = str(retrieval_query.get("text") or "").strip()
            top_k = int(retrieval_query.get("top_k") or 3)
            result = self.rag_engine.retrieve(text, filters, top_k=top_k)
            return (
                {
                    "id": retrieval_query.get("id") or "",
                    "strategy": retrieval_query.get("strategy") or "",
                    "query": text,
                    "top_k": top_k,
                    "source_count": len(result.sources),
                    "cache": (result.status or {}).get("cache") or {},
                    "channels": (result.status or {}).get("channels") or {},
                },
                result,
            )

        batch_degraded = ""
        if len(executable_queries) > 1 and hasattr(self.rag_engine, "retrieve_many"):
            try:
                retrieval_mode = "batch_multi_query_rewrite"
                results = self.rag_engine.retrieve_many(executable_queries, filters)
                for index, (retrieval_query, result) in enumerate(zip(executable_queries, results)):
                    query_statuses[index] = query_status_from_result(retrieval_query, result)
            except Exception as exc:
                batch_degraded = type(exc).__name__
                query_statuses = [None] * len(executable_queries)
                results = []
        else:
            results = []

        if not results and len(executable_queries) > 1 and self.max_retrieval_workers > 1:
            retrieval_mode = "parallel_multi_query_rewrite"
            results_by_index: list[RetrievalResult | None] = [None] * len(executable_queries)
            with ThreadPoolExecutor(max_workers=min(self.max_retrieval_workers, len(executable_queries))) as executor:
                futures = {executor.submit(retrieve_one, retrieval_query): index for index, retrieval_query in enumerate(executable_queries)}
                for future in as_completed(futures):
                    index = futures[future]
                    query_status, result = future.result()
                    query_statuses[index] = query_status
                    results_by_index[index] = result
            results = [result for result in results_by_index if result is not None]
        elif not results:
            retrieval_mode = "multi_query_rewrite"
            results = []
            for index, retrieval_query in enumerate(executable_queries):
                query_status, result = retrieve_one(retrieval_query)
                query_statuses[index] = query_status
                results.append(result)
        merged = merge_retrieval_sources(results)
        sources = sorted(
            merged.values(),
            key=lambda source: (float(source.rerank_score or 0.0), float(source.score or 0.0)),
            reverse=True,
        )[:6]
        ordered_statuses = [status for status in query_statuses if status is not None]
        status = {
            "query": "\n".join(str(query.get("text") or "") for query in executable_queries if query.get("text")),
            "query_count": len([query for query in executable_queries if query.get("text")]),
            "original_query_count": len([query for query in retrieval_queries if query.get("text")]),
            "deduped_query_count": len(executable_queries),
            "queries": ordered_statuses,
            "source_count": len(sources),
            "filter": filters,
            "metadata_filtering": bool(filters),
            "retrieval_mode": retrieval_mode,
            "retrieval_parallelism": min(self.max_retrieval_workers, max(1, len(executable_queries))),
            "batch_degraded": batch_degraded,
        }
        return RetrievalResult(sources=sources, filter=filters, status=status)

    def hybrid_retrieve(self, state: DeclarationQAState) -> DeclarationQAState:
        """执行普通知识问答混合检索。"""
        next_state = dict(state)
        result = self.multi_retrieve(next_state, next_state.get("retrieval_filter") or {})
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", {**result.status, "rag_query": next_state.get("rag_query") or ""})
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
        if is_declaration_qa_stream_run():
            stream_generate = getattr(self.answer_generator, "stream_generate", None)
        else:
            stream_generate = None
        if callable(stream_generate):
            token_stream, prompt = stream_generate(dict(next_state))
            writer = get_stream_writer()
            answer_parts = []
            for token in token_stream:
                text = str(token or "")
                if not text:
                    continue
                answer_parts.append(text)
                writer({"event": "token", "data": {"content": text}})
            answer = "".join(answer_parts)
            add_event(next_state, "answer_generated", {"intent": next_state.get("intent"), "streamed": True})
        else:
            answer, prompt = self.answer_generator.generate(next_state)
            add_event(next_state, "answer_generated", {"intent": next_state.get("intent")})
        next_state["answer"] = answer
        audit_payload = dict(next_state.get("audit_payload") or {})
        audit_payload["final_prompt"] = prompt
        next_state["audit_payload"] = audit_payload
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
            "ticket_fact_type": next_state.get("ticket_fact_type") or "none",
            "retrieval_focus": next_state.get("retrieval_focus") or "",
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
            "retrieval_query": next_state.get("retrieval_query") or "",
            "rag_query": next_state.get("rag_query") or "",
            "query_rewrite": next_state.get("query_rewrite") or {},
            "retrieval_queries": next_state.get("retrieval_queries") or [],
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
    if metadata_filter_has_item_scope(state.get("retrieval_filter") or {}):
        return query
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


def metadata_filter_has_item_scope(filters: dict[str, Any]) -> bool:
    """判断检索条件是否已具备事项级约束。"""
    return any(filters.get(key) for key in ("item_code", "ssqd_code", "ssqd_id", "item_name"))


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
    filters: dict[str, Any] = {}
    if allow_request_filter:
        filters.update(normalize_retrieval_filter(state.get("metadata_filter") or {}))
    filters.update(ticket_metadata_filter(state))
    return normalize_retrieval_filter(filters)


def ticket_metadata_filter(state: DeclarationQAState) -> dict[str, Any]:
    """从当前办件上下文生成知识库可执行的事项过滤条件。"""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return {}
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    for key in ("item_code", "ssqd_code", "ssqd_id"):
        value = first_non_empty(apply_context.get(key), payload.get(key), payload.get(camelize_metadata_key(key)))
        if value:
            return {key: value}
    item_name = first_non_empty(
        apply_context.get("item_name"),
        payload.get("official_item_name"),
        payload.get("item_name"),
        payload.get("itemName"),
        apply_context.get("title"),
        payload.get("title"),
    )
    if item_name:
        return {"item_name": item_name}
    return {}


def first_non_empty(*values: Any) -> str:
    """返回第一个非空字符串值。"""
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def camelize_metadata_key(key: str) -> str:
    """将 snake_case 元数据键转为常见 camelCase 别名。"""
    parts = key.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


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
        if (
            any(token in lowered for token in ["id_card", "idcard", "phone", "mobile"])
            or lowered in {"name", "applicant_name", "contact_name", "person_name"}
            or lowered.endswith("_address")
        ):
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
    planned_query = normalize_text(str(question.get("original_query") or question.get("query") or ""))
    if not planned_query:
        return
    state["query"] = planned_query
    retrieval_query = normalize_text(str(question.get("retrieval_query") or planned_query))
    state["normalized_query"] = retrieval_query
    state["retrieval_query"] = retrieval_query
    state["requires_ticket_context"] = bool(question.get("requires_ticket_context"))
    state["ticket_fact_type"] = question.get("ticket_fact_type") or "none"
    state["retrieval_focus"] = question.get("retrieval_focus") or classify_intent(planned_query)
    state["biz_type"] = state.get("biz_type") or infer_biz_type(retrieval_query) or infer_biz_type(planned_query)


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


def dedupe_retrieval_queries(retrieval_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按检索文本去重，保留 rewrite 原始顺序。"""
    output: list[dict[str, Any]] = []
    seen = set()
    for query in retrieval_queries:
        text = normalize_text(str(query.get("text") or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        item = dict(query)
        item["text"] = text
        output.append(item)
    return output


def query_status_from_result(retrieval_query: dict[str, Any], result: RetrievalResult) -> dict[str, Any]:
    """构造单条 rewrite query 的检索状态。"""
    return {
        "id": retrieval_query.get("id") or "",
        "strategy": retrieval_query.get("strategy") or "",
        "query": str(retrieval_query.get("text") or "").strip(),
        "top_k": int(retrieval_query.get("top_k") or 3),
        "source_count": len(result.sources),
        "cache": (result.status or {}).get("cache") or {},
        "channels": (result.status or {}).get("channels") or {},
        "batch_mode": bool((result.status or {}).get("batch_mode")),
    }


def merge_retrieval_sources(results: list[RetrievalResult]) -> dict[str, RetrievedSource]:
    """合并多路检索来源，同一 chunk 保留最高分命中。"""
    merged: dict[str, RetrievedSource] = {}
    for result in results:
        for source in result.sources:
            existing = merged.get(source.chunk.id)
            if existing is None or float(source.score or 0.0) > float(existing.score or 0.0):
                merged[source.chunk.id] = source
    return merged


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


def graph_stream_part(chunk: Any) -> tuple[str, Any]:
    """归一化 LangGraph 多模式 stream chunk。"""
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return str(chunk[0]), chunk[1]
    return "updates", chunk


def normalize_stream_event(payload: Any) -> dict[str, Any] | None:
    """将 LangGraph custom stream payload 转为前端 SSE 事件。"""
    if not isinstance(payload, dict):
        return None
    return {"event": str(payload.get("event") or "event"), "data": payload.get("data") or {}}


def should_emit_sources(node_name: str, state: DeclarationQAState) -> bool:
    """判断节点完成后是否需要补发检索来源事件。"""
    if not state.get("retrieved_sources"):
        return False
    if node_name in {"ticket_knowledge_rag", "hybrid_retrieve"}:
        return True
    return node_name == "multi_question_collect" and not state.get("need_ticket_selection")


def done_event(state: DeclarationQAState) -> dict[str, Any]:
    """构造流式响应结束事件。"""
    return {
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


def is_declaration_qa_stream_run() -> bool:
    """判断当前节点是否由 DeclarationQAWorkflow.stream 启动。"""
    try:
        config = get_config()
    except RuntimeError:
        return False
    configurable = config.get("configurable") or {}
    return bool(configurable.get("declaration_qa_stream"))


def mask_sensitive_text(value: str) -> str:
    """遮盖文本中的敏感证件和手机号信息。"""
    text = str(value or "")
    text = re.sub(r"\b\d{17}[\dXx]\b", "[id_card]", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "[phone]", text)
    return text


def is_unsafe_query(query: str) -> bool:
    """判断是否为不应回答的高风险问题。"""
    return any(keyword in str(query or "") for keyword in ["伪造", "造假", "假材料", "绕过审核"])


def task_intent(retrieval_focus: str, ticket_fact_type: str, requires_ticket_context: bool, query: str = "") -> str:
    """从 Planner 任务字段派生兼容的 intent 展示值。"""
    if requires_ticket_context:
        if ticket_fact_type == "status":
            return INTENT_TICKET_STATUS
        if ticket_fact_type == "rejection":
            return INTENT_REJECTION_REASON
        if ticket_fact_type == "correction":
            return INTENT_CORRECTION_GUIDANCE
    if retrieval_focus == "other":
        return classify_intent(query) if query else "knowledge_qa"
    return str(retrieval_focus or "knowledge_qa")


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
