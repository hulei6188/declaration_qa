"""政务服务智能体 LangGraph 工作流编排入口。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from gov_service_agent.api.schemas import ChatRequest, ChatResponse, HumanConfirmationRequest
from gov_service_agent.application.agent import ToolCallingAgent
from gov_service_agent.application.helpers.retrieval_sources import select_answer_sources
from gov_service_agent.application.helpers.streaming import (
    STREAM_TOKEN_DELAY_SECONDS,
    STREAM_TOKEN_SIZE,
    done_event,
    graph_stream_part,
    normalize_stream_event,
    should_emit_sources,
    token_events,
)
from gov_service_agent.application.nodes import ApplicationNodes
from gov_service_agent.application.nodes.agent_runtime import apply_human_confirmation_selections
from gov_service_agent.application.state import GovServiceAgentState
from gov_service_agent.application.tools import (
    ApplicationSubmitTool,
    AppointmentBookTool,
    AppointmentCancelTool,
    AppointmentHallQueryTool,
    AppointmentListQueryTool,
    AppointmentSlotQueryTool,
    ConsultationRecordQueryTool,
    ComplaintCreateTool,
    DeliveryStatusQueryTool,
    KnowledgeSearchTool,
    PaymentOrderQueryTool,
    PaymentReceiptQueryTool,
    QueueStatusQueryTool,
    ResultCertificateQueryTool,
    SubmissionReceiptQueryTool,
    TicketDetailQueryTool,
    TicketListQueryTool,
    ToolRegistry,
)
from gov_service_agent.core.business import BusinessDataAdapter, InMemoryBusinessDataAdapter
from gov_service_agent.memory import LangGraphMemoryRuntime, build_memory_resources
from gov_service_agent.observability.langfuse import create_trace_id, observation, safe_value
from gov_service_agent.observability.performance import performance_request
from gov_service_agent.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider
from gov_service_agent.rag.engine.base import RAGEngine
from gov_service_agent.rag.engine.seed import build_seed_rag_engine
from gov_service_agent.storage.audit import InMemoryAuditStore


__all__ = [
    "STREAM_TOKEN_DELAY_SECONDS",
    "STREAM_TOKEN_SIZE",
    "GovServiceAgentWorkflow",
    "select_answer_sources",
    "token_events",
]


class GovServiceAgentWorkflow(ApplicationNodes):
    """政务服务智能体的图工作流编排器。"""
    def __init__(
        self,
        *,
        rag_engine: RAGEngine | None = None,
        business_adapter: BusinessDataAdapter | None = None,
        audit_store: InMemoryAuditStore | None = None,
        agent_model: Any | None = None,
        agent_multimodal_model: Any | None = None,
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
        memory_runtime: LangGraphMemoryRuntime | None = None,
        max_retrieval_workers: int = 4,
        min_sources_per_query: int = 2,
        max_accumulated_sources: int = 48,
    ) -> None:
        """初始化政务服务智能体工作流。"""
        self.rag_engine = rag_engine or build_seed_rag_engine()
        self.business_adapter = business_adapter or InMemoryBusinessDataAdapter()
        self.audit_store = audit_store or InMemoryAuditStore()
        if agent is not None and agent_planner is not None:
            raise ValueError("agent and legacy agent_planner cannot both be provided")
        self.agent = agent or agent_planner or ToolCallingAgent(
            agent_model,
            multimodal_model=agent_multimodal_model,
            multimodal_enabled=agent_multimodal_enabled,
            allowed_image_dirs=agent_allowed_image_dirs,
            max_evidence_images=agent_max_evidence_images,
            min_image_score=agent_min_image_score,
            max_image_bytes=agent_max_image_bytes,
        )
        self.agent_max_iterations = max(1, int(agent_max_iterations or 4))
        self.query_image_ocr_client = query_image_ocr_client
        self.query_image_ocr_max_chars = max(1, int(query_image_ocr_max_chars or 4000))
        self.memory_runtime = memory_runtime or LangGraphMemoryRuntime(
            resources=build_memory_resources(
                database_url=None,
                embedding_provider=HashEmbeddingProvider(),
            ),
            model=getattr(self.agent, "model", agent_model),
            summary_max_tokens=72000,
            summary_trigger_tokens=48000,
            summary_output_tokens=5000,
            long_term_top_k=5,
            background_workers=1,
        )
        self.max_retrieval_workers = max(1, int(max_retrieval_workers))
        self.min_sources_per_query = max(0, int(min_sources_per_query or 0))
        self.max_accumulated_sources = max(1, int(max_accumulated_sources or 48))
        self.tool_registry = tool_registry or ToolRegistry()
        if self.tool_registry.get("ticket_list_query") is None:
            self.tool_registry.register(TicketListQueryTool(self.business_adapter))
        if self.tool_registry.get("ticket_detail_query") is None:
            self.tool_registry.register(TicketDetailQueryTool(self.business_adapter))
        if self.tool_registry.get("knowledge_search") is None:
            self.tool_registry.register(KnowledgeSearchTool(self.execute_knowledge_search))
        business_query_tools = [
            AppointmentListQueryTool(self.business_adapter),
            AppointmentHallQueryTool(self.business_adapter),
            AppointmentSlotQueryTool(self.business_adapter),
            AppointmentBookTool(self.business_adapter),
            AppointmentCancelTool(self.business_adapter),
            QueueStatusQueryTool(self.business_adapter),
            PaymentOrderQueryTool(self.business_adapter),
            PaymentReceiptQueryTool(self.business_adapter),
            SubmissionReceiptQueryTool(self.business_adapter),
            ResultCertificateQueryTool(self.business_adapter),
            DeliveryStatusQueryTool(self.business_adapter),
            ConsultationRecordQueryTool(self.business_adapter),
            ApplicationSubmitTool(self.business_adapter),
            ComplaintCreateTool(self.business_adapter),
        ]
        for tool in business_query_tools:
            if self.tool_registry.get(tool.name) is None:
                self.tool_registry.register(tool)
        self.graph = self._build_graph()

    def invoke(self, request: ChatRequest) -> ChatResponse:
        """同步执行问答工作流并返回最终响应。"""
        initial_state = self._initial_state(request)
        config = self._graph_config(
            user_id=int(initial_state["user_id"]),
            session_id=str(initial_state["session_id"]),
        )
        with performance_request(
            "workflow.invoke",
            session_id=str(initial_state.get("session_id") or ""),
            trace_id=str(initial_state.get("langfuse_trace_id") or ""),
            user_id=int(initial_state.get("user_id") or 0),
        ), observation(
            "workflow.invoke",
            input=chat_trace_input(request),
            metadata=chat_trace_metadata(initial_state),
            trace_id=str(initial_state.get("langfuse_trace_id") or ""),
            user_id=str(initial_state.get("user_id") or ""),
            session_id=str(initial_state.get("session_id") or ""),
            tags=["gov-service-agent", "chat"],
            trace_name="gov_service_agent.chat",
        ) as span:
            pending_confirmation = self._current_human_confirmation(config)
            if pending_confirmation:
                final_state = {
                    **initial_state,
                    "answer": "请先处理当前待确认的预约操作。",
                }
            else:
                graph_state = self.graph.invoke(initial_state, config=config)
                # LangGraph 在 interrupt（例如预约提交前的人机确认）处暂停时，
                # 某些版本只返回当前节点的增量状态。保留请求级基础字段，避免
                # 同步接口在读取 session_id 等字段时因状态不完整而返回 500。
                final_state = {**initial_state, **dict(graph_state or {})}
            pending_after_run = self._current_human_confirmation(config)
            if pending_after_run:
                observe_human_confirmation_requested(
                    pending_after_run,
                    final_state,
                    source="workflow.invoke",
                )
            span.update(output=chat_trace_output(final_state), metadata=chat_trace_metadata(final_state))
            span.set_trace_io(input=chat_trace_input(request), output=chat_trace_output(final_state))
        return ChatResponse(
            session_id=str(final_state["session_id"]),
            audit_id=str(final_state.get("audit_id") or ""),
            run_id=str(final_state.get("audit_id") or ""),
            intent=str(final_state.get("intent") or "unknown"),
            route=str(final_state.get("route") or ""),
            sub_intent=str(final_state.get("sub_intent") or ""),
            route_decision=final_state.get("route_decision") or {},
            retrieval_chain=str(final_state.get("retrieval_chain") or ""),
            answer=str(final_state.get("answer") or ("操作正在等待人工确认。" if not final_state.get("audit_id") else "")),
            refused=bool(final_state.get("refused")),
            refuse_reason=final_state.get("refuse_reason"),
            sources=final_state.get("retrieved_sources") or [],
            business_snapshot=final_state.get("business_snapshot") or {},
            rag_status=final_state.get("rag_status") or {},
            answer_mode=str(final_state.get("answer_mode") or ""),
            agent_plan=final_state.get("agent_plan") or {},
            agent_query_plan=final_state.get("agent_query_plan") or {},
            tool_results=final_state.get("tool_results") or [],
            human_confirmation=self._current_human_confirmation(config),
            dialog_task=final_state.get("dialog_task") or {},
            suggested_actions=final_state.get("suggested_actions") or [],
            events=final_state.get("events") or [],
        )

    def stream(self, request: ChatRequest) -> Iterable[dict[str, Any]]:
        """通过 LangGraph 流式执行问答工作流并产出 SSE 事件。"""
        state: GovServiceAgentState = self._initial_state(request)
        with performance_request(
            "workflow.stream",
            session_id=str(state.get("session_id") or ""),
            trace_id=str(state.get("langfuse_trace_id") or ""),
            user_id=int(state.get("user_id") or 0),
        ), observation(
            "workflow.stream",
            input=chat_trace_input(request),
            metadata=chat_trace_metadata(state),
            trace_id=str(state.get("langfuse_trace_id") or ""),
            user_id=str(state.get("user_id") or ""),
            session_id=str(state.get("session_id") or ""),
            tags=["gov-service-agent", "chat", "stream"],
            trace_name="gov_service_agent.chat.stream",
        ) as span:
            yield {"event": "run_started", "data": {"session_id": state["session_id"]}}

            config = self._graph_config(
                user_id=int(state["user_id"]),
                session_id=str(state["session_id"]),
                stream=True,
            )
            pending_confirmation = self._current_human_confirmation(config)
            if pending_confirmation:
                observe_human_confirmation_requested(
                    pending_confirmation,
                    state,
                    source="workflow.stream.pending",
                )
                yield {"event": "human_confirmation_required", "data": pending_confirmation}
                yield {
                    "event": "run_paused",
                    "data": {
                        "session_id": str(state.get("session_id") or ""),
                        "reason": "human_confirmation_required",
                    },
                }
                return
            interrupted = yield from self._stream_graph_events(
                state,
                state=state,
                config=config,
                emitted_event_count=0,
            )

            span.update(output=chat_trace_output(state), metadata=chat_trace_metadata(state))
            span.set_trace_io(input=chat_trace_input(request), output=chat_trace_output(state))
            if interrupted:
                return
            yield done_event(state)

    def resume_stream(self, request: HumanConfirmationRequest) -> Iterable[dict[str, Any]]:
        """使用可信人工决定恢复暂停的预约写操作。"""
        config = self._graph_config(
            user_id=int(request.user_id),
            session_id=str(request.session_id),
            stream=True,
        )
        snapshot = self.graph.get_state(config)
        matched_interrupt = None
        for task in snapshot.tasks or ():
            for pending in task.interrupts or ():
                if str(pending.id) == str(request.interrupt_id):
                    matched_interrupt = pending
                    break
        if matched_interrupt is None:
            raise ValueError("人工确认已失效、已处理或不属于当前会话")

        state: GovServiceAgentState = dict(snapshot.values or {})
        if int(state.get("user_id") or 0) != int(request.user_id):
            raise ValueError("人工确认不属于当前用户")
        if request.approved:
            apply_human_confirmation_selections(
                list(state.get("agent_pending_tool_calls") or []),
                state,
                request.selected_slot_ids,
            )
        trace_id = str(state.get("langfuse_trace_id") or "")
        resume_input = human_confirmation_trace_input(request)
        with performance_request(
            "workflow.resume_stream",
            session_id=str(request.session_id or ""),
            trace_id=trace_id,
            user_id=int(request.user_id or 0),
        ), observation(
            "workflow.resume_stream",
            input=resume_input,
            metadata={**chat_trace_metadata(state), "human_confirmation": resume_input},
            trace_id=trace_id,
            user_id=str(request.user_id or ""),
            session_id=str(request.session_id or ""),
            tags=["gov-service-agent", "chat", "resume", "human-confirmation"],
            trace_name="gov_service_agent.chat",
        ) as span:
            observe_human_confirmation_resolved(request, state)
            yield {
                "event": "run_resumed",
                "data": {
                    "session_id": request.session_id,
                    "interrupt_id": request.interrupt_id,
                    "approved": request.approved,
                },
            }
            interrupted = yield from self._stream_graph_events(
                Command(
                    resume={
                        "approved": bool(request.approved),
                        "interrupt_id": str(request.interrupt_id),
                        "selected_slot_ids": dict(request.selected_slot_ids),
                    }
                ),
                state=state,
                config=config,
                emitted_event_count=len(state.get("events") or []),
            )
            span.update(
                output={
                    **chat_trace_output(state),
                    "human_confirmation": {
                        "approved": bool(request.approved),
                        "interrupt_id": str(request.interrupt_id),
                    },
                },
                metadata=chat_trace_metadata(state),
            )
            if interrupted:
                return
            yield done_event(state)

    def _current_human_confirmation(self, config: dict[str, Any]) -> dict[str, Any]:
        """读取当前线程尚未处理的人工确认。"""
        snapshot = self.graph.get_state(config)
        for task in snapshot.tasks or ():
            for pending in task.interrupts or ():
                value = dict(pending.value or {}) if isinstance(pending.value, dict) else {}
                return {
                    **value,
                    "interrupt_id": str(pending.id),
                    "session_id": str((snapshot.values or {}).get("session_id") or ""),
                }
        return {}

    def _stream_graph_events(
        self,
        graph_input: GovServiceAgentState | Command,
        *,
        state: GovServiceAgentState,
        config: dict[str, Any],
        emitted_event_count: int,
    ) -> Iterable[dict[str, Any]]:
        """统一处理首次运行和人工确认恢复后的 LangGraph 事件。"""
        streamed_answer = False
        for chunk in self.graph.stream(graph_input, config=config, stream_mode=["updates", "custom"]):
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
                if node_name == "__interrupt__":
                    for pending in node_state or ():
                        value = dict(pending.value or {}) if isinstance(pending.value, dict) else {}
                        confirmation = {
                            **value,
                            "interrupt_id": str(pending.id),
                            "session_id": str(state.get("session_id") or ""),
                        }
                        observe_human_confirmation_requested(
                            confirmation,
                            state,
                            source="workflow.stream",
                        )
                        yield {"event": "human_confirmation_required", "data": confirmation}
                    yield {
                        "event": "run_paused",
                        "data": {
                            "session_id": str(state.get("session_id") or ""),
                            "reason": "human_confirmation_required",
                        },
                    }
                    return True
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
                elif node_name == "commit_agent_answer" and not streamed_answer:
                    yield from token_events(str(state.get("answer") or ""))
        return False

    def _initial_state(self, request: ChatRequest) -> GovServiceAgentState:
        """根据请求构造工作流初始状态。"""
        session_id = request.session_id or uuid4().hex
        checkpoint_state = self._checkpoint_state(
            user_id=int(request.user_id),
            session_id=session_id,
        )
        return {
            "user_id": request.user_id,
            "session_id": session_id,
            "query": request.query,
            "normalized_query": "",
            "query_image": request.query_image.model_dump(exclude={"data_url"}) if request.query_image else None,
            "form_submission": request.form_submission.model_dump() if request.form_submission else None,
            "query_image_ocr_text": "",
            "query_image_ocr_status": {},
            "long_term_context": {},
            "memory_context": dict(checkpoint_state.get("memory_context") or {}),
            "conversation_memory": dict(checkpoint_state.get("conversation_memory") or {}),
            "working_memory": {},
            "dialog_task": dict(checkpoint_state.get("dialog_task") or {}),
            "agent_iteration": 0,
            "agent_next": "",
            "agent_plan": {},
            "agent_query_plan": {},
            "agent_pending_tool_calls": [],
            "human_confirmation": {},
            "agent_call_signatures": [],
            "agent_messages": list(checkpoint_state.get("agent_messages") or []),
            "agent_input_messages": [],
            "answer_mode": "",
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
            "evidence_registry": {"knowledge": [], "business": []},
            "active_evidence_registry": {"knowledge": [], "business": []},
            "refuse_reason": None,
            "answer": "",
            "refused": False,
            "suggested_actions": [],
            "events": [],
            "audit_payload": {},
            "langfuse_trace_id": create_trace_id(f"{session_id}:{request.query}"),
        }

    def _checkpoint_state(self, *, user_id: int, session_id: str) -> dict[str, Any]:
        """仅从 LangGraph 检查点存储加载先前的线程状态。"""
        snapshot = self.graph.get_state(
            self._graph_config(user_id=user_id, session_id=session_id)
        )
        return dict(snapshot.values or {})

    def get_session(self, *, user_id: int, session_id: str) -> dict[str, Any] | None:
        """加载历史消息，并把最新多轮表单状态补到最后一条助手消息。"""
        item = self.audit_store.get_session(user_id=user_id, session_id=session_id)
        if not item:
            return None
        config = self._graph_config(user_id=user_id, session_id=session_id)
        try:
            checkpoint = self._checkpoint_state(user_id=user_id, session_id=session_id)
            pending_confirmation = self._current_human_confirmation(config)
        except Exception:
            checkpoint = {}
            pending_confirmation = {}
        dialog_task = checkpoint.get("dialog_task") if isinstance(checkpoint.get("dialog_task"), dict) else {}
        if dialog_task.get("status") == "collecting" or pending_confirmation:
            messages = [dict(message) for message in (item.get("messages") or [])]
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    if dialog_task.get("status") == "collecting":
                        message["dialog_task"] = dialog_task
                    if pending_confirmation:
                        message["confirmation"] = {**pending_confirmation, "status": "pending"}
                    break
            item = {**item, "messages": messages}
        return item

    def _graph_config(self, *, user_id: int, session_id: str, stream: bool = False) -> dict[str, Any]:
        """构造线程配置，并为完整 Agent 轮次预留足够的图节点步数。"""
        return {
            "recursion_limit": max(32, self.agent_max_iterations * 4 + 16),
            "configurable": {
                "thread_id": f"user:{int(user_id)}:session:{session_id}",
                "langgraph_user_id": str(int(user_id)),
                "gov_service_agent_stream": bool(stream),
            }
        }

    def close(self) -> None:
        """关闭后台记忆任务和 PostgreSQL 持久化资源。"""
        self.memory_runtime.close()
        disposed: set[int] = set()
        for store in getattr(self, "managed_stores", []):
            engine = getattr(store, "engine", None)
            if engine is None or id(engine) in disposed:
                continue
            disposed.add(id(engine))
            engine.dispose()

    def delete_session(self, *, user_id: int, session_id: str) -> bool:
        """删除产品历史记录及其对应的 LangGraph 线程状态。"""
        deleter = getattr(self.audit_store, "delete_session", None)
        if not callable(deleter):
            return False
        deleted = bool(deleter(user_id=user_id, session_id=session_id))
        if deleted:
            thread_id = self._graph_config(
                user_id=user_id,
                session_id=session_id,
            )["configurable"]["thread_id"]
            self.memory_runtime.checkpointer.delete_thread(thread_id)
        return deleted

    def _build_graph(self):
        """构建受控 Agent Tool-Calling 工作流。"""
        builder = StateGraph(GovServiceAgentState)
        builder.add_node("context_build", self.context_build)
        builder.add_node("query_image_enrich", self.query_image_enrich)
        builder.add_node("query_preprocess", self.query_preprocess)
        builder.add_node("agent_prepare", self.agent_prepare)
        builder.add_node("agent_summarize", self.agent_summarize)
        builder.add_node("agent_refresh_input", self.agent_refresh_input)
        builder.add_node("agent_decide", self.agent_decide)
        builder.add_node("agent_execute_tools", self.agent_execute_tools)
        builder.add_node("agent_observe", self.agent_observe)
        builder.add_node("refuse_or_human", self.refuse_or_human)
        builder.add_node("commit_agent_answer", self.commit_agent_answer)
        builder.add_node("answer_postprocess", self.answer_postprocess)
        builder.add_node("audit_log", self.audit_log)

        builder.add_edge(START, "context_build")
        builder.add_edge("context_build", "query_image_enrich")
        builder.add_edge("query_image_enrich", "query_preprocess")
        builder.add_edge("query_preprocess", "agent_prepare")
        builder.add_edge("agent_prepare", "agent_summarize")
        builder.add_edge("agent_summarize", "agent_decide")
        builder.add_conditional_edges(
            "agent_decide",
            self._agent_decision_route,
            {
                "tools": "agent_execute_tools",
                "commit": "commit_agent_answer",
                "refuse": "refuse_or_human",
            },
        )
        builder.add_edge("agent_execute_tools", "agent_observe")
        builder.add_edge("agent_observe", "agent_refresh_input")
        builder.add_edge("agent_refresh_input", "agent_decide")
        builder.add_edge("commit_agent_answer", "answer_postprocess")
        builder.add_edge("refuse_or_human", "answer_postprocess")
        builder.add_edge("answer_postprocess", "audit_log")
        builder.add_edge("audit_log", END)
        return builder.compile(
            checkpointer=self.memory_runtime.checkpointer,
            store=self.memory_runtime.store,
        )

    def _agent_decision_route(self, state: GovServiceAgentState) -> str:
        """Tool Call 继续循环，无 Tool Call 的文本直接作为最终回答。"""
        if state.get("agent_next") == "tools":
            return "tools"
        if state.get("agent_final_answer"):
            return "commit"
        return "refuse"

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


def human_confirmation_trace_input(request: HumanConfirmationRequest) -> dict[str, Any]:
    """构建不包含业务正文的人工确认观测输入。"""
    return {
        "session_id": str(request.session_id or ""),
        "interrupt_id": str(request.interrupt_id or ""),
        "approved": bool(request.approved),
        "selected_slot_ids": {
            str(call_id): int(slot_id)
            for call_id, slot_id in (request.selected_slot_ids or {}).items()
        },
    }


def observe_human_confirmation_requested(
    confirmation: dict[str, Any],
    state: dict[str, Any],
    *,
    source: str,
) -> None:
    """在当前聊天 Trace 中记录一次等待人工确认事件。"""
    actions = list(confirmation.get("actions") or [])
    trace_id = str(state.get("langfuse_trace_id") or "")
    payload = {
        "interrupt_id": str(confirmation.get("interrupt_id") or ""),
        "action_count": len(actions),
        "tools": [str(action.get("tool_name") or "") for action in actions],
        "source": source,
    }
    with observation(
        "agent.human_confirmation.requested",
        as_type="span",
        input=payload,
        metadata={
            **chat_trace_metadata(state),
            "human_confirmation_status": "requested",
            "interrupt_id": payload["interrupt_id"],
        },
        trace_id=trace_id,
    ) as span:
        span.update(output={"status": "awaiting_confirmation", **payload})


def observe_human_confirmation_resolved(
    request: HumanConfirmationRequest,
    state: dict[str, Any],
) -> None:
    """在恢复执行前记录人工确认结果。"""
    trace_id = str(state.get("langfuse_trace_id") or "")
    payload = human_confirmation_trace_input(request)
    with observation(
        "agent.human_confirmation.resolved",
        as_type="span",
        input=payload,
        metadata={
            **chat_trace_metadata(state),
            "human_confirmation_status": "approved" if request.approved else "rejected",
            "interrupt_id": str(request.interrupt_id or ""),
        },
        trace_id=trace_id,
    ) as span:
        span.update(
            output={
                "status": "approved" if request.approved else "rejected",
                "interrupt_id": str(request.interrupt_id or ""),
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
            "answer_mode": state.get("answer_mode") or "",
            "has_query_image": bool(state.get("query_image")),
            "agent_message_count": len(state.get("agent_messages") or []),
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
            "long_term_context": long_term_trace_metrics(state.get("long_term_context") or {}),
            "rag_status": state.get("rag_status") or {},
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
