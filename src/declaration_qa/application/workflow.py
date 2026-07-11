"""申报问答 LangGraph 工作流编排入口。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from declaration_qa.api.schemas import ChatRequest, ChatResponse
from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.agent import ToolCallingAgent
from declaration_qa.application.tools import KnowledgeSearchTool, TicketDetailQueryTool, TicketListQueryTool, ToolRegistry
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
from declaration_qa.application.workflow_nodes import WorkflowNodes
from declaration_qa.core.business import BusinessDataAdapter, InMemoryBusinessDataAdapter
from declaration_qa.memory.manager import ContextMemoryManager
from declaration_qa.observability.langfuse import create_trace_id, observation, safe_value
from declaration_qa.memory.context import RECENT_MESSAGE_LIMIT, recent_messages_from_session
from declaration_qa.rag.engine.base import RAGEngine
from declaration_qa.rag.engine.seed import build_seed_rag_engine
from declaration_qa.rag.prompting import DeterministicAnswerGenerator
from declaration_qa.storage.audit import InMemoryAuditStore


__all__ = [
    "STREAM_TOKEN_DELAY_SECONDS",
    "STREAM_TOKEN_SIZE",
    "DeclarationQAWorkflow",
    "select_answer_sources",
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
        agent_model: Any | None = None,
        agent: Any | None = None,
        agent_planner: Any | None = None,
        agent_multimodal_enabled: bool = False,
        agent_allowed_image_dirs: list[str] | None = None,
        agent_max_evidence_images: int = 3,
        agent_min_image_score: float = 0.7,
        agent_max_image_bytes: int = 10 * 1024 * 1024,
        tool_registry: ToolRegistry | None = None,
        agent_max_iterations: int = 4,
        query_image_ocr_client: Any | None = None,
        query_image_ocr_max_chars: int = 4000,
        context_memory_manager: ContextMemoryManager | None = None,
        max_retrieval_workers: int = 4,
    ) -> None:
        """初始化申报问答工作流。"""
        self.rag_engine = rag_engine or build_seed_rag_engine()
        self.business_adapter = business_adapter or InMemoryBusinessDataAdapter()
        self.audit_store = audit_store or InMemoryAuditStore()
        self.answer_generator = answer_generator or DeterministicAnswerGenerator()
        if agent is not None and agent_planner is not None:
            raise ValueError("agent and legacy agent_planner cannot both be provided")
        self.agent = agent or agent_planner or ToolCallingAgent(
            agent_model,
            multimodal_enabled=agent_multimodal_enabled,
            allowed_image_dirs=agent_allowed_image_dirs,
            max_evidence_images=agent_max_evidence_images,
            min_image_score=agent_min_image_score,
            max_image_bytes=agent_max_image_bytes,
        )
        self.agent_max_iterations = max(1, int(agent_max_iterations or 4))
        self.query_image_ocr_client = query_image_ocr_client
        self.query_image_ocr_max_chars = max(1, int(query_image_ocr_max_chars or 4000))
        self.context_memory_manager = context_memory_manager
        self.max_retrieval_workers = max(1, int(max_retrieval_workers))
        self.tool_registry = tool_registry or ToolRegistry()
        if self.tool_registry.get("ticket_list_query") is None:
            self.tool_registry.register(TicketListQueryTool(self.business_adapter))
        if self.tool_registry.get("ticket_detail_query") is None:
            self.tool_registry.register(TicketDetailQueryTool(self.business_adapter))
        if self.tool_registry.get("knowledge_search") is None:
            self.tool_registry.register(KnowledgeSearchTool(self.execute_knowledge_search))
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
            run_id=str(final_state["audit_id"]),
            intent=str(final_state.get("intent") or "unknown"),
            route=str(final_state.get("route") or ""),
            sub_intent=str(final_state.get("sub_intent") or ""),
            route_decision=final_state.get("route_decision") or {},
            retrieval_chain=str(final_state.get("retrieval_chain") or ""),
            answer=str(final_state.get("answer") or ""),
            refused=bool(final_state.get("refused")),
            refuse_reason=final_state.get("refuse_reason"),
            sources=final_state.get("retrieved_sources") or [],
            business_snapshot=final_state.get("business_snapshot") or {},
            rag_status=final_state.get("rag_status") or {},
            evidence_decision=final_state.get("evidence_decision") or {},
            agent_plan=final_state.get("agent_plan") or {},
            agent_query_plan=final_state.get("agent_query_plan") or {},
            tool_results=final_state.get("tool_results") or [],
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
                    elif node_name == "agent_finalize_answer" and not streamed_answer:
                        yield from token_events(str(state.get("answer") or ""))

            span.update(output=chat_trace_output(state), metadata=chat_trace_metadata(state))
            span.set_trace_io(input=chat_trace_input(request), output=chat_trace_output(state))
            yield done_event(state)

    def _initial_state(self, request: ChatRequest) -> DeclarationQAState:
        """根据请求构造工作流初始状态。"""
        session_id = request.session_id or uuid4().hex
        fallback_messages = self._recent_session_messages(request)
        short_memory = self._load_short_memory(request, session_id=session_id, fallback_messages=fallback_messages)
        session_messages = list(short_memory.get("recent_messages") or fallback_messages)
        return {
            "user_id": request.user_id,
            "session_id": session_id,
            "query": request.query,
            "normalized_query": "",
            "query_image": request.query_image.model_dump(exclude={"data_url"}) if request.query_image else None,
            "query_image_ocr_text": "",
            "query_image_ocr_status": {},
            "session_messages": session_messages,
            "short_memory": short_memory,
            "short_term_context": {},
            "long_term_context": {},
            "agent_iteration": 0,
            "agent_next": "",
            "agent_plan": {},
            "agent_query_plan": {},
            "agent_pending_tool_calls": [],
            "agent_call_signatures": [],
            "agent_required_evidence": [],
            "agent_final_answer": "",
            "agent_final_prompt": "",
            "agent_answer_provider": "",
            "agent_final_model": "",
            "agent_final_token_usage": {},
            "agent_final_latency_ms": None,
            "frozen_knowledge_query": "",
            "agent_selected_item_code": "",
            "agent_selected_application_id": None,
            "tool_results": [],
            "region_code": request.region_code,
            "app_version": request.app_version,
            "policy_version": request.policy_version,
            "biz_type": request.biz_type,
            "request_biz_type": request.biz_type,
            "metadata_filter": request.metadata_filter,
            "intent": "unknown",
            "sub_intent": "",
            "route": "",
            "route_decision": {},
            "retrieval_plan": {},
            "retrieval_chain": "",
            "retrieval_query": "",
            "rag_query": "",
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

    def _load_short_memory(
        self,
        request: ChatRequest,
        *,
        session_id: str,
        fallback_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """从 Redis 加载短期上下文记忆，异常时回退到会话消息。"""
        if self.context_memory_manager is None:
            return {"recent_messages": fallback_messages, "summary": "", "backend": "none", "cache_hit": False}
        try:
            return self.context_memory_manager.load_short_memory(
                user_id=int(request.user_id),
                session_id=session_id,
                fallback_messages=fallback_messages,
            )
        except Exception:
            return {"recent_messages": fallback_messages, "summary": "", "backend": "error", "cache_hit": False}

    def _recent_session_messages(self, request: ChatRequest) -> list[dict[str, Any]]:
        """读取当前会话最近消息，避免把全量历史送入模型。"""
        session_id = str(request.session_id or "").strip()
        if not session_id:
            return []
        try:
            recent_messages = getattr(self.audit_store, "recent_messages", None)
            if callable(recent_messages):
                return list(
                    recent_messages(
                        user_id=int(request.user_id),
                        session_id=session_id,
                        limit=RECENT_MESSAGE_LIMIT,
                    )
                    or []
                )
            session = self.audit_store.get_session(user_id=int(request.user_id), session_id=session_id)
            return recent_messages_from_session(session, limit=RECENT_MESSAGE_LIMIT)
        except Exception:
            return []

    def _build_graph(self):
        """构建受控 Agent Tool-Calling 工作流。"""
        builder = StateGraph(DeclarationQAState)
        builder.add_node("context_build", self.context_build)
        builder.add_node("query_image_enrich", self.query_image_enrich)
        builder.add_node("query_preprocess", self.query_preprocess)
        builder.add_node("agent_prepare", self.agent_prepare)
        builder.add_node("agent_decide", self.agent_decide)
        builder.add_node("agent_execute_tools", self.agent_execute_tools)
        builder.add_node("agent_observe", self.agent_observe)
        builder.add_node("agent_evidence_check", self.agent_evidence_check)
        builder.add_node("refuse_or_human", self.refuse_or_human)
        builder.add_node("agent_finalize_answer", self.agent_finalize_answer)
        builder.add_node("answer_postprocess", self.answer_postprocess)
        builder.add_node("audit_log", self.audit_log)

        builder.add_edge(START, "context_build")
        builder.add_edge("context_build", "query_image_enrich")
        builder.add_edge("query_image_enrich", "query_preprocess")
        builder.add_edge("query_preprocess", "agent_prepare")
        builder.add_edge("agent_prepare", "agent_decide")
        builder.add_conditional_edges(
            "agent_decide",
            lambda state: state.get("agent_next") or "finish",
            {"tools": "agent_execute_tools", "finish": "agent_evidence_check"},
        )
        builder.add_edge("agent_execute_tools", "agent_observe")
        builder.add_edge("agent_observe", "agent_evidence_check")
        builder.add_conditional_edges(
            "agent_evidence_check",
            self._agent_evidence_route,
            {
                "ok": "agent_finalize_answer",
                "retry": "agent_decide",
                "refuse": "refuse_or_human",
            },
        )
        builder.add_edge("agent_finalize_answer", "answer_postprocess")
        builder.add_edge("refuse_or_human", "answer_postprocess")
        builder.add_edge("answer_postprocess", "audit_log")
        builder.add_edge("audit_log", END)
        return builder.compile()

    def _agent_evidence_route(self, state: DeclarationQAState) -> str:
        """Route complete evidence directly to one final model call."""
        iteration = int(state.get("agent_iteration") or 0)
        if state.get("evidence_ok"):
            can_finalize = callable(getattr(self.agent, "finalize", None))
            if can_finalize or state.get("agent_final_answer") or iteration >= self.agent_max_iterations:
                return "ok"
            return "retry"
        if state.get("refuse_reason") or iteration >= self.agent_max_iterations:
            return "refuse"
        return "retry"

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
            "has_query_image": request.query_image is not None,
            "query_image": request.query_image.model_dump(exclude={"data_url"}) if request.query_image else None,
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
            "retrieval_chain": state.get("retrieval_chain") or "",
            "agent_iteration": int(state.get("agent_iteration") or 0),
            "tool_result_count": len(state.get("tool_results") or []),
            "has_query_image": bool(state.get("query_image")),
            "recent_message_count": len(state.get("session_messages") or []),
            "short_term_context": ((state.get("short_term_context") or {}).get("metrics") or {}),
            "long_term_context": long_term_trace_metrics(state.get("long_term_context") or {}),
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
            "agent_iteration": int(state.get("agent_iteration") or 0),
            "tool_result_count": len(state.get("tool_results") or []),
            "source_count": len(state.get("retrieved_sources") or []),
            "short_term_context": ((state.get("short_term_context") or {}).get("metrics") or {}),
            "long_term_context": long_term_trace_metrics(state.get("long_term_context") or {}),
            "rag_status": state.get("rag_status") or {},
            "evidence_decision": state.get("evidence_decision") or {},
            "suggested_actions": state.get("suggested_actions") or [],
        }
    )


def long_term_trace_metrics(context: dict[str, Any]) -> dict[str, Any]:
    """构建长期记忆 trace 指标。"""
    items = context.get("items") if isinstance(context, dict) else []
    status = context.get("status") if isinstance(context, dict) else {}
    cache = context.get("cache") if isinstance(context, dict) else {}
    return {
        "memory_count": len(items or []) if isinstance(items, list) else 0,
        "retrieval_mode": (status or {}).get("retrieval_mode") if isinstance(status, dict) else "",
        "cache_hit": bool((cache or {}).get("hit")) if isinstance(cache, dict) else False,
    }
