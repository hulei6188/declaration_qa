"""上下文记忆格式化工具。"""

from __future__ import annotations

import json
from typing import Any


RECENT_TURN_LIMIT = 5
FIELD_CONTENT_LIMIT = 300
EVIDENCE_OVERVIEW_LIMIT = 6


def build_short_term_context(state: dict[str, Any]) -> dict[str, Any]:
    """构建答案生成使用的短期上下文。"""
    short_memory = state.get("short_memory") if isinstance(state.get("short_memory"), dict) else {}
    context = {
        "current_question": current_question_context(state),
        "business_tool_context": business_tool_context_summary(state),
        "query_image": query_image_summary(state.get("query_image")),
        "evidence_overview": evidence_overview(state.get("retrieved_sources") or []),
    }
    context["metrics"] = {
        "agent_message_count": len(state.get("agent_messages") or []),
        "short_memory_backend": short_memory.get("backend") or "",
        "short_memory_cache_hit": bool(short_memory.get("cache_hit")),
        "has_business_tool_context": bool(context["business_tool_context"]),
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


def compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """移除空值，保留布尔和数字。"""
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        output[key] = value
    return output


def truncate_text(value: Any, limit: int) -> str:
    """截断文本字段。"""
    text = str(value or "").strip()
    max_length = max(0, int(limit or 0))
    if max_length <= 0 or len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."
