"""Agent 历史对话与当前轮证据的隔离策略。"""

from __future__ import annotations

import json
from typing import Any


RECENT_CONVERSATION_USER_TURNS = 3
HISTORICAL_ASSISTANT_PREFIX = "【历史助手回答，仅用于理解指代，不得作为事实证据】"
HISTORICAL_TOOL_AUDIT_KEY = "historical_tool_audit"
MAX_HISTORICAL_TOOL_CALLS_PER_TURN = 24


def retain_historical_tool_audit(
    messages: list[dict[str, Any]],
    memory_context: dict[str, Any] | None,
    *,
    keep_recent_user_turns: int = RECENT_CONVERSATION_USER_TURNS,
) -> dict[str, Any]:
    """保留最近已完成轮次的最小工具审计信息，不保留参数或结果内容。"""
    context = dict(memory_context or {})
    valid_messages = [dict(item) for item in messages if isinstance(item, dict)]
    last_user_index = next(
        (
            index
            for index in range(len(valid_messages) - 1, -1, -1)
            if str(valid_messages[index].get("role") or "") == "user"
        ),
        None,
    )
    if last_user_index is None:
        return context

    user_message = valid_messages[last_user_index]
    turn_id = str(user_message.get("id") or "").strip()
    if not turn_id:
        return context

    result_by_call_id: dict[str, dict[str, Any]] = {}
    for item in valid_messages[last_user_index + 1 :]:
        if str(item.get("role") or "") != "tool":
            continue
        payload = _tool_payload(item.get("content"))
        call_id = str(item.get("tool_call_id") or payload.get("call_id") or "").strip()
        if call_id:
            result_by_call_id[call_id] = payload

    tools: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    for item in valid_messages[last_user_index + 1 :]:
        if str(item.get("role") or "") != "assistant":
            continue
        for call in item.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "").strip()
            tool_name = str(call.get("name") or "").strip()[:160]
            if not tool_name or (call_id and call_id in seen_call_ids):
                continue
            if call_id:
                seen_call_ids.add(call_id)
            result = result_by_call_id.get(call_id, {})
            audit_item: dict[str, Any] = {
                "tool_name": tool_name,
                "status": str(result.get("status") or "requested")[:80],
            }
            if "ok" in result:
                audit_item["ok"] = bool(result.get("ok"))
            tools.append(audit_item)
            if len(tools) >= MAX_HISTORICAL_TOOL_CALLS_PER_TURN:
                break
        if len(tools) >= MAX_HISTORICAL_TOOL_CALLS_PER_TURN:
            break

    records = [
        dict(item)
        for item in context.get(HISTORICAL_TOOL_AUDIT_KEY) or []
        if isinstance(item, dict) and str(item.get("turn_id") or "").strip()
    ]
    current_record = {"turn_id": turn_id, "tools": tools}
    records = [item for item in records if str(item.get("turn_id") or "") != turn_id]
    records.append(current_record)
    keep_count = max(0, int(keep_recent_user_turns))
    context[HISTORICAL_TOOL_AUDIT_KEY] = records[-keep_count:] if keep_count else []
    return context


def _tool_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def project_historical_messages(
    messages: list[dict[str, Any]],
    *,
    keep_recent_user_turns: int = RECENT_CONVERSATION_USER_TURNS,
    memory_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """保留对话语义，但从新一轮模型输入中移除全部历史工具消息。"""
    del memory_context
    valid_messages = [dict(item) for item in messages if isinstance(item, dict)]
    user_indexes = [
        index
        for index, item in enumerate(valid_messages)
        if str(item.get("role") or "") == "user"
    ]
    keep_count = max(0, int(keep_recent_user_turns))
    if keep_count and len(user_indexes) > keep_count:
        recent_start = user_indexes[-keep_count]
    elif keep_count:
        recent_start = 0
    else:
        recent_start = len(valid_messages)

    output: list[dict[str, Any]] = []
    for index, copied in enumerate(valid_messages):
        role = str(copied.get("role") or "")
        if role == "tool":
            continue
        if role == "assistant":
            if copied.get("tool_calls"):
                continue
            if index < recent_start:
                continue
            content = str(copied.get("content") or "").strip()
            if content:
                if content.startswith(HISTORICAL_ASSISTANT_PREFIX):
                    content = content[len(HISTORICAL_ASSISTANT_PREFIX) :].strip()
                copied["content"] = f"{HISTORICAL_ASSISTANT_PREFIX}{content[:1200]}"
        output.append(copied)
    return output


def build_conversation_memory(memory_context: dict[str, Any] | None) -> dict[str, Any]:
    """构造只用于指代解析的滚动摘要上下文。"""
    context = memory_context or {}
    running_summary = context.get("running_summary")
    summary_text = getattr(running_summary, "summary", "")
    if isinstance(running_summary, dict):
        summary_text = str(running_summary.get("summary") or "")
    audit_records = [
        item
        for item in context.get(HISTORICAL_TOOL_AUDIT_KEY) or []
        if isinstance(item, dict)
    ]
    audit_count = len(audit_records)
    historical_tool_audit = [
        {
            "relative_turn": audit_count - index,
            "tools": [
                {
                    key: tool[key]
                    for key in ("tool_name", "status", "ok")
                    if key in tool
                }
                for tool in item.get("tools") or []
                if isinstance(tool, dict) and str(tool.get("tool_name") or "").strip()
            ],
        }
        for index, item in enumerate(audit_records)
    ]
    return {
        "running_summary": str(summary_text or ""),
        "assistant_history_trust": "conversation_only",
        "knowledge_evidence_retention": "current_turn_only",
        "business_tool_history_window": 0,
        "historical_tool_audit": historical_tool_audit,
        "historical_tool_audit_note": (
            "relative_turn=1 表示上一用户轮次；仅证明工具调用及执行状态，"
            "不包含参数、结果或可复用事实证据。"
        ),
    }
