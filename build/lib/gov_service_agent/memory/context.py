"""上下文记忆格式化工具。"""

from __future__ import annotations

import json
from typing import Any


RECENT_TURN_LIMIT = 5
RECENT_MESSAGE_LIMIT = RECENT_TURN_LIMIT * 2
MESSAGE_CONTENT_LIMIT = 500
FIELD_CONTENT_LIMIT = 300
EVIDENCE_OVERVIEW_LIMIT = 6


def recent_messages_from_session(session: dict[str, Any] | None, *, limit: int = RECENT_MESSAGE_LIMIT) -> list[dict[str, Any]]:
    """从会话详情中裁剪最近消息。"""
    if not isinstance(session, dict):
        return []
    return normalize_recent_messages(session.get("messages") or [], limit=limit)


def normalize_recent_messages(messages: list[dict[str, Any]], *, limit: int = RECENT_MESSAGE_LIMIT) -> list[dict[str, Any]]:
    """裁剪并清理最近消息。"""
    output = [normalize_message(message) for message in list(messages or [])]
    output = [message for message in output if message.get("content") or message.get("attachments")]
    return output[-max(0, int(limit or RECENT_MESSAGE_LIMIT)) :]


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """清理单条历史消息。"""
    payload = message if isinstance(message, dict) else {}
    item: dict[str, Any] = {
        "role": str(payload.get("role") or ""),
        "content": truncate_text(payload.get("content") or "", MESSAGE_CONTENT_LIMIT),
    }
    if payload.get("intent"):
        item["intent"] = truncate_text(payload.get("intent") or "", 80)
    if payload.get("refused"):
        item["refused"] = True
    attachments = normalize_attachments(payload.get("attachments") or [])
    if attachments:
        item["attachments"] = attachments
    return item


def build_short_term_context(state: dict[str, Any]) -> dict[str, Any]:
    """构建答案生成使用的短期上下文。"""
    short_memory = state.get("short_memory") if isinstance(state.get("short_memory"), dict) else {}
    recent_messages = normalize_recent_messages(short_memory.get("recent_messages") or state.get("session_messages") or [])
    context = {
        "current_question": current_question_context(state),
        "summary": truncate_text(short_memory.get("summary") or "", 1200),
        "recent_messages": recent_messages,
        "ticket_context": ticket_context_summary(state),
        "query_image": query_image_summary(state.get("query_image")),
        "evidence_overview": evidence_overview(state.get("retrieved_sources") or []),
    }
    context["metrics"] = {
        "recent_message_count": len(recent_messages),
        "recent_turn_limit": RECENT_TURN_LIMIT,
        "short_memory_backend": short_memory.get("backend") or "",
        "short_memory_cache_hit": bool(short_memory.get("cache_hit")),
        "has_short_memory_summary": bool(context.get("summary")),
        "has_ticket_context": bool(context["ticket_context"]),
        "has_query_image": bool(context["query_image"]),
        "source_count": len(state.get("retrieved_sources") or []),
        "evidence_overview_count": len(context["evidence_overview"]),
    }
    return context


def format_short_term_context(state: dict[str, Any]) -> str:
    """格式化短期上下文，供提示词使用。"""
    context = state.get("short_term_context")
    if not isinstance(context, dict) or not context:
        context = build_short_term_context(state)
    return json.dumps(context, ensure_ascii=False, indent=2)


def format_long_term_context(state: dict[str, Any]) -> str:
    """格式化长期上下文记忆，供提示词使用。"""
    context = state.get("long_term_context") if isinstance(state.get("long_term_context"), dict) else {}
    if not context:
        context = {"items": [], "status": {"retrieval_mode": "none", "reason": "not_loaded"}}
    return json.dumps(context, ensure_ascii=False, indent=2)


def current_question_context(state: dict[str, Any]) -> dict[str, Any]:
    """构建当前问题上下文。"""
    return compact_dict(
        {
            "query": truncate_text(state.get("query") or "", FIELD_CONTENT_LIMIT),
            "normalized_query": truncate_text(state.get("normalized_query") or "", FIELD_CONTENT_LIMIT),
            "intent": state.get("intent") or "",
            "route": state.get("route") or "",
            "retrieval_chain": state.get("retrieval_chain") or "",
        }
    )


def ticket_context_summary(state: dict[str, Any]) -> dict[str, Any]:
    """整理当前工单上下文摘要。"""
    ticket_context = state.get("ticket_context") if isinstance(state.get("ticket_context"), dict) else {}
    ticket = ticket_context.get("ticket") if isinstance(ticket_context.get("ticket"), dict) else {}
    apply_context = state.get("apply_context") if isinstance(state.get("apply_context"), dict) else {}
    business_snapshot = state.get("business_snapshot") if isinstance(state.get("business_snapshot"), dict) else {}
    reject_info = state.get("reject_info") if isinstance(state.get("reject_info"), dict) else {}
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    return compact_dict(
        {
            "apply_id": first_non_empty(ticket.get("ticket_id"), apply_context.get("apply_id"), state.get("apply_id")),
            "biz_type": first_non_empty(ticket.get("biz_type"), apply_context.get("biz_type"), state.get("biz_type")),
            "title": truncate_text(
                first_non_empty(
                    ticket.get("title"),
                    apply_context.get("title"),
                    apply_context.get("item_name"),
                    payload.get("official_item_name"),
                    payload.get("item_name"),
                ),
                FIELD_CONTENT_LIMIT,
            ),
            "status": truncate_text(first_non_empty(ticket.get("status"), apply_context.get("status")), 120),
            "node_name": truncate_text(
                first_non_empty(
                    ticket.get("node_name"),
                    ticket.get("current_node"),
                    apply_context.get("node_name"),
                    apply_context.get("current_node"),
                    business_snapshot.get("node_name"),
                    business_snapshot.get("current_node"),
                ),
                120,
            ),
            "reject_reason": truncate_text(first_non_empty(reject_info.get("reject_reason"), business_snapshot.get("reject_reason")), FIELD_CONTENT_LIMIT),
            "material_name": truncate_text(first_non_empty(reject_info.get("material_name"), business_snapshot.get("material_name")), 120),
        }
    )


def query_image_summary(query_image: dict[str, Any] | None) -> dict[str, Any]:
    """整理当前上传图片摘要。"""
    if not isinstance(query_image, dict) or not query_image:
        return {}
    return compact_dict(
        {
            "filename": query_image.get("filename") or "",
            "content_type": query_image.get("content_type") or "",
            "size_bytes": query_image.get("size_bytes") or 0,
            "text_present": query_image.get("text_present"),
            "has_file_path": bool(str(query_image.get("file_path") or "").strip()),
        }
    )


def evidence_overview(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成证据概览，完整证据仍由 source_payload 提供。"""
    output = []
    for source in list(sources or [])[:EVIDENCE_OVERVIEW_LIMIT]:
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        output.append(
            compact_dict(
                {
                    "title": truncate_text(source.get("title") or "", 120),
                    "chunk_id": truncate_text(source.get("chunk_id") or "", 160),
                    "score": source.get("score"),
                    "rerank_score": source.get("rerank_score"),
                    "chunk_type": metadata.get("chunk_type") or metadata.get("semantic_chunk_type") or "",
                    "section_path": truncate_text(metadata.get("section_path") or "", FIELD_CONTENT_LIMIT),
                    "node_title": truncate_text(metadata.get("node_title") or "", 160),
                    "attachment_name": truncate_text(metadata.get("attachment_name") or "", 160),
                    "attachment_page_no": metadata.get("attachment_page_no"),
                }
            )
        )
    return output


def normalize_attachments(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """清理消息附件摘要。"""
    output = []
    for attachment in list(attachments or [])[:3]:
        if not isinstance(attachment, dict):
            continue
        output.append(
            compact_dict(
                {
                    "filename": truncate_text(attachment.get("filename") or "", 160),
                    "content_type": attachment.get("content_type") or "",
                    "size_bytes": attachment.get("size_bytes") or 0,
                    "url": attachment.get("url") or "",
                }
            )
        )
    return output


def compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """移除空值，保留布尔和数字。"""
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        output[key] = value
    return output


def first_non_empty(*values: Any) -> str:
    """返回第一个非空字符串。"""
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def truncate_text(value: Any, limit: int) -> str:
    """截断文本字段。"""
    text = str(value or "").strip()
    max_length = max(0, int(limit or 0))
    if max_length <= 0 or len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."
