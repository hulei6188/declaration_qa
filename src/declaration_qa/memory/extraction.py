"""长期上下文记忆抽取。"""

from __future__ import annotations

import hashlib
from typing import Any

from declaration_qa.memory.models import ContextMemoryItem, utc_now_text


MAX_QUERY_CHARS = 220
MAX_ANSWER_CHARS = 900
MAX_SOURCE_TITLES = 3


class ContextMemoryExtractor:
    """从一轮问答审计载荷中抽取长期记忆。"""

    def extract(self, payload: dict[str, Any], *, audit_id: str) -> list[ContextMemoryItem]:
        """抽取可长期复用的记忆条目。"""
        query = compact_text(payload.get("query") or "", MAX_QUERY_CHARS)
        answer = compact_text(payload.get("answer") or "", MAX_ANSWER_CHARS)
        if not query or not answer:
            return []
        if bool(payload.get("refused")) and str(payload.get("refuse_reason") or "") not in {"clarification_required"}:
            return []
        if has_business_tool_result(payload):
            return []
        source_summary = summarize_sources(payload.get("retrieved_sources") or [])
        content_parts = [
            f"用户问题：{query}",
            f"系统回答摘要：{answer}",
        ]
        if source_summary:
            content_parts.append(f"关联办事指南或材料：{source_summary}")
        content = "\n".join(content_parts)
        memory_type = infer_memory_type(payload)
        memory_id = stable_memory_id(
            user_id=int(payload.get("user_id") or 0),
            memory_type=memory_type,
            content=content,
        )
        return [
            ContextMemoryItem(
                id=memory_id,
                user_id=int(payload.get("user_id") or 0),
                session_id=str(payload.get("session_id") or ""),
                memory_type=memory_type,
                content=content,
                summary=answer[:240],
                source_audit_id=audit_id,
                source_query=query,
                source_answer=answer,
                region_code=str(payload.get("region_code") or ""),
                biz_type=str(payload.get("biz_type") or ""),
                importance=memory_importance(payload),
                metadata={
                    "intent": payload.get("intent") or "",
                    "route": payload.get("route") or "",
                    "retrieval_chain": payload.get("retrieval_chain") or "",
                    "source_count": len(payload.get("retrieved_sources") or []),
                },
                created_at=utc_now_text(),
            )
        ]


def infer_memory_type(payload: dict[str, Any]) -> str:
    """根据问答链路推断记忆类型。"""
    intent = str(payload.get("intent") or "")
    if payload.get("query_image"):
        return "image_query_fact"
    return "qa_fact"


def memory_importance(payload: dict[str, Any]) -> float:
    """估算记忆重要性，供后续排序使用。"""
    score = 0.45
    if payload.get("retrieved_sources"):
        score += 0.15
    if str(payload.get("intent") or "") in {"material", "process_steps", "time_limit", "status", "reject"}:
        score += 0.1
    return min(1.0, round(score, 3))


def summarize_sources(sources: list[dict[str, Any]]) -> str:
    """整理来源标题和事项线索。"""
    labels = []
    for source in list(sources or [])[:MAX_SOURCE_TITLES]:
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        title = str(source.get("title") or "").strip()
        item_name = str(metadata.get("item_name") or source.get("item_name") or "").strip()
        section = str(metadata.get("section_path") or metadata.get("node_title") or "").strip()
        label = " / ".join(part for part in [title, item_name, section] if part)
        if label:
            labels.append(label[:180])
    return "；".join(dict.fromkeys(labels))


def has_business_tool_result(payload: dict[str, Any]) -> bool:
    """实时个人工单事实禁止写入可跨轮复用的长期记忆。"""
    return any(
        isinstance(result, dict) and str(result.get("tool_name") or "").startswith("ticket_")
        for result in payload.get("tool_results") or []
    )


def stable_memory_id(*, user_id: int, memory_type: str, content: str) -> str:
    """根据内容生成稳定记忆 ID，避免重复写入。"""
    digest = hashlib.sha256(f"{user_id}:{memory_type}:{content}".encode("utf-8")).hexdigest()[:32]
    return f"mem_{digest}"


def compact_text(value: Any, limit: int) -> str:
    """压缩空白并截断文本。"""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
