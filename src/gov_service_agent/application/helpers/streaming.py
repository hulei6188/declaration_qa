"""LangGraph 流式输出辅助函数。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from langgraph.config import get_config

from gov_service_agent.application.state import GovServiceAgentState


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


def should_emit_sources(node_name: str, state: GovServiceAgentState) -> bool:
    """判断节点完成后是否需要补发检索来源事件。"""
    if not state.get("retrieved_sources"):
        return False
    return node_name == "agent_observe"


def done_event(state: GovServiceAgentState) -> dict[str, Any]:
    """构造流式响应结束事件。"""
    return {
        "event": "done",
        "data": {
            "session_id": state.get("session_id"),
            "audit_id": state.get("audit_id"),
            "run_id": state.get("audit_id"),
            "intent": state.get("intent"),
            "route": state.get("route"),
            "sub_intent": state.get("sub_intent"),
            "retrieval_chain": state.get("retrieval_chain"),
            "agent_plan": state.get("agent_plan") or {},
            "agent_query_plan": state.get("agent_query_plan") or {},
            "agent_iteration": int(state.get("agent_iteration") or 0),
            "selected_item_code": state.get("agent_selected_item_code") or "",
            "selected_application_id": state.get("agent_selected_application_id"),
            "dialog_task": state.get("dialog_task") or {},
            "refused": bool(state.get("refused")),
            "refuse_reason": state.get("refuse_reason"),
            "suggested_actions": state.get("suggested_actions") or [],
        },
    }


def is_gov_service_agent_stream_run() -> bool:
    """判断当前节点是否由 GovServiceAgentWorkflow.stream 启动。"""
    try:
        config = get_config()
    except RuntimeError:
        return False
    configurable = config.get("configurable") or {}
    return bool(configurable.get("gov_service_agent_stream"))
