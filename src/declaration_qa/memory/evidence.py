"""Agent 历史对话与当前轮证据的隔离策略。"""

from __future__ import annotations

from typing import Any


RECENT_CONVERSATION_USER_TURNS = 3
HISTORICAL_ASSISTANT_PREFIX = "【历史助手回答，仅用于理解指代，不得作为事实证据】"


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
    running_summary = (memory_context or {}).get("running_summary")
    summary_text = getattr(running_summary, "summary", "")
    if isinstance(running_summary, dict):
        summary_text = str(running_summary.get("summary") or "")
    return {
        "running_summary": str(summary_text or ""),
        "assistant_history_trust": "conversation_only",
        "knowledge_evidence_retention": "current_turn_only",
        "business_tool_history_window": 0,
    }
