"""Agent 知识证据与业务工具上下文的分层保留策略。"""

from __future__ import annotations

from typing import Any


RECENT_EVIDENCE_USER_TURNS = 3
HISTORICAL_ASSISTANT_PREFIX = "【历史助手回答，仅用于理解指代，不得作为事实证据】"
RETROSPECTIVE_TERMS = ("刚才", "刚刚", "之前", "先前", "上一轮", "上次", "当时", "前面显示")
CURRENT_STATE_TERMS = ("现在", "当前", "目前", "最新", "实时", "是否已经", "有没有变化")


def project_historical_messages(
    messages: list[dict[str, Any]],
    *,
    keep_recent_user_turns: int = RECENT_EVIDENCE_USER_TURNS,
    memory_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """知识原文保留到被摘要覆盖，业务结果仅保留最近若干用户轮次。"""
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

    summarized_ids, summarized_boundary = summarized_message_scope(
        valid_messages,
        memory_context,
    )
    retained_tool_call_ids: set[str] = set()
    for index, item in enumerate(valid_messages):
        if str(item.get("role") or "") != "tool":
            continue
        call_id = str(item.get("tool_call_id") or "")
        if not call_id or is_reference_only_tool_observation(item.get("content")):
            continue
        message_id = str(item.get("id") or "")
        is_summarized = (
            bool(message_id and message_id in summarized_ids)
            or (summarized_boundary >= 0 and index <= summarized_boundary)
        )
        if index >= recent_start or (
            is_knowledge_tool_observation(item) and not is_summarized
        ):
            retained_tool_call_ids.add(call_id)

    output: list[dict[str, Any]] = []
    for index, copied in enumerate(valid_messages):
        role = str(copied.get("role") or "")
        if role == "assistant":
            tool_calls = [
                dict(call)
                for call in copied.get("tool_calls") or []
                if isinstance(call, dict) and str(call.get("id") or "") in retained_tool_call_ids
            ]
            if copied.get("tool_calls"):
                if not tool_calls:
                    continue
                copied["content"] = ""
                copied["tool_calls"] = tool_calls
            else:
                if index < recent_start:
                    continue
                content = str(copied.get("content") or "").strip()
                if content:
                    if content.startswith(HISTORICAL_ASSISTANT_PREFIX):
                        content = content[len(HISTORICAL_ASSISTANT_PREFIX) :].strip()
                    copied["content"] = f"{HISTORICAL_ASSISTANT_PREFIX}{content[:1200]}"
        elif role == "tool":
            call_id = str(copied.get("tool_call_id") or "")
            if call_id not in retained_tool_call_ids:
                continue
        output.append(copied)
    return output


def is_knowledge_tool_observation(item: dict[str, Any]) -> bool:
    """判断工具消息是否包含知识检索的完整结果。"""
    payload = item.get("content")
    if not isinstance(payload, dict):
        return str(item.get("name") or "") == "knowledge_search"
    return str(payload.get("tool_name") or item.get("name") or "") == "knowledge_search"


def summarized_message_scope(
    messages: list[dict[str, Any]],
    memory_context: dict[str, Any] | None,
) -> tuple[set[str], int]:
    """读取滚动摘要已覆盖的消息标识和兜底边界。"""
    running_summary = (memory_context or {}).get("running_summary")
    summarized_ids = getattr(running_summary, "summarized_message_ids", set())
    last_message_id = getattr(running_summary, "last_summarized_message_id", None)
    if isinstance(running_summary, dict):
        summarized_ids = running_summary.get("summarized_message_ids") or set()
        last_message_id = running_summary.get("last_summarized_message_id")
    normalized_ids = {
        str(message_id)
        for message_id in summarized_ids or []
        if str(message_id or "")
    }
    boundary = -1
    if last_message_id:
        for index, item in enumerate(messages):
            if str(item.get("id") or "") == str(last_message_id):
                boundary = index
                break
    return normalized_ids, boundary


def is_reference_only_tool_observation(value: Any) -> bool:
    """识别旧版本留下的纯引用工具结果，防止它继续进入三轮窗口。"""
    if not isinstance(value, dict):
        return False
    if str(value.get("memory_trust") or "") == "reference_only":
        return True
    data = value.get("data") if isinstance(value.get("data"), dict) else {}
    return bool(data.get("source_refs")) and not bool(data.get("sources"))


def recent_tool_context(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从窗口内完整 ToolMessage 中恢复知识正文和可回顾的业务结果。"""
    knowledge_sources: list[dict[str, Any]] = []
    business_results: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict) or str(item.get("role") or "") != "tool":
            continue
        payload = item.get("content")
        if not isinstance(payload, dict) or not payload.get("ok"):
            continue
        tool_name = str(payload.get("tool_name") or item.get("name") or "")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if tool_name == "knowledge_search":
            for source in data.get("sources") or []:
                if not isinstance(source, dict) or not str(source.get("chunk_id") or "").strip():
                    continue
                remembered = dict(source)
                remembered["memory_origin"] = "recent_tool_context"
                knowledge_sources = upsert_by_key(knowledge_sources, remembered, key="chunk_id")
            continue
        remembered_result = dict(payload)
        remembered_result["memory_origin"] = "historical_snapshot"
        remembered_result["observed_at"] = str(
            payload.get("completed_at") or payload.get("observed_at") or ""
        )
        business_results = upsert_by_key(
            business_results,
            remembered_result,
            key="call_id",
        )
    return knowledge_sources, business_results


def build_conversation_memory(memory_context: dict[str, Any] | None) -> dict[str, Any]:
    """构造只用于指代解析的滚动摘要上下文。"""
    running_summary = (memory_context or {}).get("running_summary")
    summary_text = getattr(running_summary, "summary", "")
    if isinstance(running_summary, dict):
        summary_text = str(running_summary.get("summary") or "")
    return {
        "running_summary": str(summary_text or ""),
        "assistant_history_trust": "conversation_only",
        "knowledge_evidence_retention": "until_rolling_summary",
        "business_tool_history_window": RECENT_EVIDENCE_USER_TURNS,
    }


def is_retrospective_query(query: str) -> bool:
    """判断用户是否明确询问最近工具结果的历史快照。"""
    text = str(query or "")
    if any(term in text for term in CURRENT_STATE_TERMS):
        return False
    return any(term in text for term in RETROSPECTIVE_TERMS)


def upsert_by_key(
    items: list[dict[str, Any]],
    value: dict[str, Any],
    *,
    key: str,
) -> list[dict[str, Any]]:
    """按稳定标识去重，并让较新的完整结果覆盖旧结果。"""
    identity = str(value.get(key) or "")
    if not identity:
        return [*items, value]
    return [
        *[item for item in items if str(item.get(key) or "") != identity],
        value,
    ]
