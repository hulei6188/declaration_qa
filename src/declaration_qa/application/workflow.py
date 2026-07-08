"""申报问答 LangGraph 工作流编排入口。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from declaration_qa.api.schemas import ChatRequest, ChatResponse
from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.workflow_helpers.events import add_event
from declaration_qa.application.workflow_helpers.retrieval_sources import select_answer_sources
from declaration_qa.application.workflow_helpers.streaming import (
    STREAM_TOKEN_DELAY_SECONDS,
    STREAM_TOKEN_SIZE,
    done_event,
    graph_stream_part,
    normalize_stream_event,
    should_emit_sources,
    token_events,
)
from declaration_qa.application.workflow_helpers.ticket_context import ticket_rag_query
from declaration_qa.application.workflow_nodes import WorkflowNodes
from declaration_qa.application.workflow_retrieval_nodes import is_image_only_query
from declaration_qa.core.business import BusinessDataAdapter, InMemoryBusinessDataAdapter
from declaration_qa.observability.langfuse import create_trace_id, observation, safe_value
from declaration_qa.rag.engine.base import RAGEngine
from declaration_qa.rag.engine.seed import build_seed_rag_engine
from declaration_qa.rag.prompting import DeterministicAnswerGenerator
from declaration_qa.rag.question_resolver.rule import RuleQuestionResolver
from declaration_qa.rag.query_rewriter.rule import RuleQueryRewriter
from declaration_qa.rag.router import (
    ROUTE_CLARIFICATION,
    ROUTE_REFUSE,
    ROUTE_TICKET_CONTEXT_RELATED,
    ROUTE_TICKET_CONTEXT_UNRELATED,
)
from declaration_qa.storage.audit import InMemoryAuditStore


__all__ = [
    "STREAM_TOKEN_DELAY_SECONDS",
    "STREAM_TOKEN_SIZE",
    "DeclarationQAWorkflow",
    "select_answer_sources",
    "ticket_rag_query",
    "token_events",
]


class DeclarationQAWorkflow(WorkflowNodes):
    """申报问答的图工作流编排器。"""
    def __init__(
        self,
        *,
        rag_engine: RAGEngine | None = None,
        business_adapter: BusinessDataAdapter | None = None,
        audit_store: InMemoryAuditStore | None = None,
        answer_generator: DeterministicAnswerGenerator | None = None,
        question_resolver: RuleQuestionResolver | None = None,
        query_rewriter: RuleQueryRewriter | None = None,
        max_sub_question_workers: int = 4,
        max_retrieval_workers: int = 4,
    ) -> None:
        """初始化申报问答工作流。"""
        self.rag_engine = rag_engine or build_seed_rag_engine()
        self.business_adapter = business_adapter or InMemoryBusinessDataAdapter()
        self.audit_store = audit_store or InMemoryAuditStore()
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        self.question_resolver = question_resolver or RuleQuestionResolver()
        self.query_rewriter = query_rewriter or RuleQueryRewriter()
        self.max_sub_question_workers = max(1, int(max_sub_question_workers))
        self.max_retrieval_workers = max(1, int(max_retrieval_workers))
        self.sub_question_graph = self._build_sub_question_graph()
        self.graph = self._build_graph()

    def invoke(self, request: ChatRequest) -> ChatResponse:
        """同步执行问答工作流并返回最终响应。"""
        initial_state = self._initial_state(request)
        with observation(
            "workflow.invoke",
            input=chat_trace_input(request),
            metadata=chat_trace_metadata(initial_state),
            trace_id=str(initial_state.get("langfuse_trace_id") or ""),
            user_id=str(initial_state.get("user_id") or ""),
            session_id=str(initial_state.get("session_id") or ""),
            tags=["declaration-qa", "chat"],
            trace_name="declaration_qa.chat",
        ) as span:
            final_state = self.graph.invoke(initial_state)
            span.update(output=chat_trace_output(final_state), metadata=chat_trace_metadata(final_state))
            span.set_trace_io(input=chat_trace_input(request), output=chat_trace_output(final_state))
        return ChatResponse(
            session_id=str(final_state["session_id"]),
            audit_id=str(final_state["audit_id"]),
            intent=str(final_state.get("intent") or "unknown"),
            route=str(final_state.get("route") or ""),
            sub_intent=str(final_state.get("sub_intent") or ""),
            route_decision=final_state.get("route_decision") or {},
            multi_question=bool(final_state.get("multi_question")),
            question_resolution=final_state.get("question_resolution") or {},
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
        with observation(
            "workflow.stream",
            input=chat_trace_input(request),
            metadata=chat_trace_metadata(state),
            trace_id=str(state.get("langfuse_trace_id") or ""),
            user_id=str(state.get("user_id") or ""),
            session_id=str(state.get("session_id") or ""),
            tags=["declaration-qa", "chat", "stream"],
            trace_name="declaration_qa.chat.stream",
        ) as span:
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

            span.update(output=chat_trace_output(state), metadata=chat_trace_metadata(state))
            span.set_trace_io(input=chat_trace_input(request), output=chat_trace_output(state))
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
        session_id = request.session_id or uuid4().hex
        return {
            "user_id": request.user_id,
            "session_id": session_id,
            "query": request.query,
            "normalized_query": "",
            "query_image": request.query_image.model_dump(exclude={"data_url"}) if request.query_image else None,
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
            "question_resolution": {},
            "requires_ticket_context": False,
            "ticket_fact_type": "none",
            "retrieval_plan": {},
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
            "langfuse_trace_id": create_trace_id(f"{session_id}:{request.query}"),
        }

    def _build_graph(self):
        """构建 LangGraph 节点和边。"""
        builder = StateGraph(DeclarationQAState)
        builder.add_node("context_build", self.context_build)
        builder.add_node("query_preprocess", self.query_preprocess)
        builder.add_node("image_only_prepare", self.image_only_prepare)
        builder.add_node("question_resolution", self.question_resolution)
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
        builder.add_conditional_edges(
            "query_preprocess",
            lambda state: "image" if is_image_only_query(state) else "text",
            {"image": "image_only_prepare", "text": "question_resolution"},
        )
        builder.add_edge("image_only_prepare", "knowledge_filter_build")
        builder.add_conditional_edges(
            "question_resolution",
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
        builder.add_conditional_edges(
            "knowledge_filter_build",
            lambda state: "image" if is_image_only_query(state) else "text",
            {"image": "hybrid_retrieve", "text": "query_rewrite"},
        )
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

    def _build_sub_question_graph(self):
        """构建多问题拆分后单个子问题的 LangGraph 子图。"""
        builder = StateGraph(DeclarationQAState)
        builder.add_node("prepare_task", self.prepare_task)
        builder.add_node("ticket_list_query", self.ticket_list_query)
        builder.add_node("ticket_selection", self.ticket_selection)
        builder.add_node("ticket_detail_query", self.ticket_detail_query)
        builder.add_node("ticket_context_build", self.ticket_context_build)
        builder.add_node("knowledge_filter_build", self.knowledge_filter_build)
        builder.add_node("query_rewrite", self.query_rewrite)
        builder.add_node("ticket_knowledge_rag", self.ticket_knowledge_rag)
        builder.add_node("hybrid_retrieve", self.hybrid_retrieve)
        builder.add_node("evidence_check", self.evidence_check)
        builder.add_node("refuse_or_human", self.refuse_or_human)

        builder.add_edge(START, "prepare_task")
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
            {"ok": END, "refuse": "refuse_or_human"},
        )
        builder.add_edge("refuse_or_human", END)
        return builder.compile()


def chat_trace_input(request: ChatRequest) -> dict[str, Any]:
    """根据聊天请求构建脱敏后的 trace 输入。"""
    return safe_value(
        {
            "user_id": request.user_id,
            "session_id": request.session_id or "",
            "query": request.query,
            "region_code": request.region_code,
            "app_version": request.app_version,
            "policy_version": request.policy_version,
            "biz_type": request.biz_type,
            "apply_id": request.apply_id,
            "has_query_image": request.query_image is not None,
            "query_image": request.query_image.model_dump(exclude={"data_url"}) if request.query_image else None,
            "has_apply_context": request.apply_context is not None,
            "metadata_filter": request.metadata_filter,
        }
    )


def chat_trace_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """根据工作流状态构建脱敏后的 trace 元数据。"""
    rag_status = state.get("rag_status") or {}
    return safe_value(
        {
            "langfuse_trace_id": state.get("langfuse_trace_id") or "",
            "user_id": state.get("user_id"),
            "session_id": state.get("session_id") or "",
            "region_code": state.get("region_code") or "",
            "app_version": state.get("app_version") or "",
            "policy_version": state.get("policy_version") or "",
            "biz_type": state.get("biz_type") or "",
            "route": state.get("route") or "",
            "intent": state.get("intent") or "",
            "sub_intent": state.get("sub_intent") or "",
            "ticket_fact_type": state.get("ticket_fact_type") or "none",
            "retrieval_chain": state.get("retrieval_chain") or "",
            "multi_question": bool(state.get("multi_question")),
            "has_query_image": bool(state.get("query_image")),
            "need_ticket_selection": bool(state.get("need_ticket_selection")),
            "refused": bool(state.get("refused")),
            "source_count": len(state.get("retrieved_sources") or []),
            "rag_retrieval_mode": rag_status.get("retrieval_mode") or "",
            "audit_id": state.get("audit_id") or "",
        }
    )


def chat_trace_output(state: dict[str, Any]) -> dict[str, Any]:
    """根据工作流状态构建脱敏后的 trace 输出。"""
    return safe_value(
        {
            "audit_id": state.get("audit_id") or "",
            "answer": state.get("answer") or "",
            "refused": bool(state.get("refused")),
            "refuse_reason": state.get("refuse_reason"),
            "route": state.get("route") or "",
            "intent": state.get("intent") or "",
            "multi_question": bool(state.get("multi_question")),
            "source_count": len(state.get("retrieved_sources") or []),
            "rag_status": state.get("rag_status") or {},
            "evidence_decision": state.get("evidence_decision") or {},
            "suggested_actions": state.get("suggested_actions") or [],
        }
    )
