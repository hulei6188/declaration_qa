"""Planner and routing nodes for the declaration QA workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.workflow_helpers.collections import unique_strings, unique_tickets
from declaration_qa.application.workflow_helpers.events import add_event
from declaration_qa.application.workflow_helpers.intent import (
    classify_intent,
    infer_biz_type,
    is_unsafe_query,
    mask_sensitive_text,
    task_intent,
)
from declaration_qa.application.workflow_helpers.planning import apply_planned_question
from declaration_qa.application.workflow_helpers.retrieval_sources import unique_sources
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, TICKET_AUGMENTED_CHAIN
from declaration_qa.rag.router import (
    INTENT_REJECTION_REASON,
    INTENT_TICKET_STATUS,
    ROUTE_CLARIFICATION,
    ROUTE_REFUSE,
    ROUTE_TICKET_CONTEXT_RELATED,
    ROUTE_TICKET_CONTEXT_UNRELATED,
)


class WorkflowPlanningNodes:
    """Context, planner, and route preparation nodes."""
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
                "sub_question_runner": "langgraph",
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
        state = self.sub_question_graph.invoke(state)
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
        """根据 Planner 任务字段准备展示字段和链路开关。"""
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
