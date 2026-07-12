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
        "verified_ticket_facts": normalize_verified_facts(
            short_memory.get("verified_ticket_facts") or short_memory.get("verified_facts") or []
        ),
        "ticket_item_overviews": normalize_ticket_item_overviews(short_memory.get("ticket_item_overviews") or []),
        "recent_knowledge_evidence_refs": normalize_knowledge_evidence_refs(
            short_memory.get("recent_knowledge_evidence_refs") or []
        ),
        "business_tool_context": business_tool_context_summary(state),
        "query_image": query_image_summary(state.get("query_image")),
        "evidence_overview": evidence_overview(state.get("retrieved_sources") or []),
    }
    context["metrics"] = {
        "recent_message_count": len(recent_messages),
        "recent_turn_limit": RECENT_TURN_LIMIT,
        "short_memory_backend": short_memory.get("backend") or "",
        "short_memory_cache_hit": bool(short_memory.get("cache_hit")),
        "has_short_memory_summary": bool(context.get("summary")),
        "verified_ticket_fact_count": len(context["verified_ticket_facts"]),
        "ticket_item_overview_count": len(context["ticket_item_overviews"]),
        "knowledge_evidence_ref_count": len(context["recent_knowledge_evidence_refs"]),
        "has_business_tool_context": bool(context["business_tool_context"]),
        "has_query_image": bool(context["query_image"]),
        "source_count": len(state.get("retrieved_sources") or []),
        "evidence_overview_count": len(context["evidence_overview"]),
    }
    return context


def normalize_verified_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅传递短期缓存中带可信来源标记的精简工单事实。"""
    output = []
    for fact in list(facts or [])[-8:]:
        if not isinstance(fact, dict) or fact.get("source") != "ticket_detail_query" or not fact.get("trusted"):
            continue
        output.append(
            compact_dict(
                {
                    "source": "ticket_detail_query",
                    "fact_type": fact.get("fact_type") or "application_status",
                    "trusted": True,
                    "item_code": truncate_text(fact.get("item_code") or "", 160),
                    "item_name": truncate_text(fact.get("item_name") or "", 160),
                    "application_id": fact.get("application_id"),
                    "current_status": truncate_text(fact.get("current_status") or "", 80),
                    "reason": truncate_text(fact.get("reason") or "", FIELD_CONTENT_LIMIT),
                    "material_name": truncate_text(fact.get("material_name") or "", 160),
                    "fact_updated_at": fact.get("fact_updated_at") or "",
                    "verified_at": fact.get("verified_at") or "",
                    "refresh_required_for_current_status": True,
                }
            )
        )
    return output


def normalize_ticket_item_overviews(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """向 Agent 提供事项级概览，但明确不能把它当成具体申请状态。"""
    output = []
    for item in list(items or [])[-8:]:
        if not isinstance(item, dict) or item.get("source") != "ticket_list_query":
            continue
        output.append(
            compact_dict(
                {
                    "source": "ticket_list_query",
                    "item_code": truncate_text(item.get("item_code") or "", 160),
                    "item_name": truncate_text(item.get("item_name") or "", 160),
                    "application_count": item.get("application_count"),
                    "statuses": list(item.get("statuses") or [])[:8],
                    "fact_updated_at": item.get("fact_updated_at") or "",
                    "verified_at": item.get("verified_at") or "",
                    "refresh_required": True,
                    "usage": "仅用于事项指代消歧，具体状态、进度和退回原因必须查询详情工具",
                }
            )
        )
    return output


def normalize_knowledge_evidence_refs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """向 Agent 暴露会话证据别名和摘要，不暴露可伪造的内部 chunk id。"""
    output = []
    for item in list(items or [])[-8:]:
        if not isinstance(item, dict) or not str(item.get("evidence_ref") or ""):
            continue
        output.append(
            compact_dict(
                {
                    "evidence_ref": truncate_text(item.get("evidence_ref") or "", 80),
                    "query": truncate_text(item.get("query") or "", FIELD_CONTENT_LIMIT),
                    "title": truncate_text(item.get("title") or "", 200),
                    "section": truncate_text(item.get("section") or "", FIELD_CONTENT_LIMIT),
                    "summary": truncate_text(item.get("summary") or "", 320),
                    "policy_version": truncate_text(item.get("policy_version") or "", 80),
                    "retrieved_at": item.get("retrieved_at") or "",
                    "expires_at": item.get("expires_at") or "",
                }
            )
        )
    return output


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


def business_tool_context_summary(state: dict[str, Any]) -> dict[str, Any]:
    """整理本轮 Agent 工具产生的业务摘要，不形成跨轮工单绑定。"""
    business_snapshot = state.get("business_snapshot") if isinstance(state.get("business_snapshot"), dict) else {}
    return compact_dict(
        {
            "source": business_snapshot.get("source") or "",
            "item_code": business_snapshot.get("item_code") or "",
            "selected_application_id": business_snapshot.get("selected_application_id"),
            "application_count": len(business_snapshot.get("applications") or []),
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
