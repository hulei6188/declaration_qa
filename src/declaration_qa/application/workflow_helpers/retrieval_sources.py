"""检索查询去重、来源合并和答案来源选择。"""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import RetrievalResult, RetrievedSource
from declaration_qa.rag.engine.structured_context import (
    structured_backfill_chunk_type,
    structured_source_matches,
)
from declaration_qa.rag.engine.text import normalize_text, safe_int


def unique_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 chunk/doc 去重来源。"""
    seen = set()
    output = []
    for source in sources:
        key = (source.get("chunk_id") or "", source.get("doc_id") or "", source.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        output.append(source)
    return output


def dedupe_retrieval_queries(retrieval_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按检索文本去重，保留 rewrite 原始顺序。"""
    output: list[dict[str, Any]] = []
    seen = set()
    for query in retrieval_queries:
        text = normalize_text(str(query.get("text") or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        item = dict(query)
        item["text"] = text
        output.append(item)
    return output


def query_status_from_result(retrieval_query: dict[str, Any], result: RetrievalResult) -> dict[str, Any]:
    """构造单条 rewrite query 的检索状态。"""
    return {
        "id": retrieval_query.get("id") or "",
        "strategy": retrieval_query.get("strategy") or "",
        "query": str(retrieval_query.get("text") or "").strip(),
        "top_k": int(retrieval_query.get("top_k") or 3),
        "source_count": len(result.sources),
        "cache": (result.status or {}).get("cache") or {},
        "channels": (result.status or {}).get("channels") or {},
        "score_cutoff": (result.status or {}).get("score_cutoff") or {},
        "batch_mode": bool((result.status or {}).get("batch_mode")),
    }


def merge_retrieval_sources(results: list[RetrievalResult]) -> dict[str, RetrievedSource]:
    """合并多路检索来源，同一 chunk 保留最高分命中。"""
    merged: dict[str, RetrievedSource] = {}
    for result in results:
        for source in result.sources:
            existing = merged.get(source.chunk.id)
            if existing is None or float(source.score or 0.0) > float(existing.score or 0.0):
                merged[source.chunk.id] = source
    return merged


def select_answer_sources(merged: dict[str, RetrievedSource], query: str, *, limit: int = 6) -> list[RetrievedSource]:
    """选择进入答案生成的来源，优先保全结构化章节节点。"""
    ranked = sorted(
        merged.values(),
        key=lambda source: (float(source.rerank_score or 0.0), float(source.score or 0.0)),
        reverse=True,
    )
    target_chunk_type = structured_backfill_chunk_type(query)
    primary_section_id = primary_structured_section_id(ranked, target_chunk_type)
    if not primary_section_id:
        return ranked[:limit]

    selected: list[RetrievedSource] = []
    seen_chunk_ids: set[str] = set()
    section_intro = [
        source
        for source in ranked
        if source_parent_section_id(source) == primary_section_id
        and source_chunk_type(source) == "section_leaf"
        and "说明" in source_node_title(source)
    ]
    for source in section_intro[:1]:
        append_source_once(selected, source, seen_chunk_ids)

    structured_sources = [
        source
        for source in ranked
        if source_parent_section_id(source) == primary_section_id
        and source_chunk_type(source) == target_chunk_type
    ]
    for source in sorted(structured_sources, key=structured_source_sort_key):
        append_source_once(selected, source, seen_chunk_ids)

    target_limit = max(limit, len(selected))
    for source in ranked:
        if len(selected) >= target_limit:
            break
        append_source_once(selected, source, seen_chunk_ids)
    return selected


def primary_structured_section_id(sources: list[RetrievedSource], target_chunk_type: str) -> str:
    """按排序后的命中结果选择最相关的结构化章节。"""
    if not target_chunk_type:
        return ""
    for source in sources:
        parent_section_id = source_parent_section_id(source)
        if parent_section_id and structured_source_matches(source, target_chunk_type):
            return parent_section_id
    return ""


def append_source_once(output: list[RetrievedSource], source: RetrievedSource, seen_chunk_ids: set[str]) -> bool:
    """按 chunk id 追加来源。"""
    if source.chunk.id in seen_chunk_ids:
        return False
    seen_chunk_ids.add(source.chunk.id)
    output.append(source)
    return True


def structured_source_sort_key(source: RetrievedSource) -> tuple[int, int, int, str]:
    """按结构化节点在原指南中的顺序排序。"""
    metadata = source.chunk.metadata or {}
    return (
        safe_int(metadata.get("node_order")),
        safe_int(metadata.get("parent_window_index")),
        safe_int(metadata.get("child_window_index")),
        source.chunk.id,
    )


def source_parent_section_id(source: RetrievedSource) -> str:
    """读取来源所属结构化章节 ID。"""
    return str((source.chunk.metadata or {}).get("parent_section_id") or "")


def source_chunk_type(source: RetrievedSource) -> str:
    """读取来源的结构化片段类型。"""
    return str((source.chunk.metadata or {}).get("chunk_type") or "")


def source_node_title(source: RetrievedSource) -> str:
    """读取来源的结构化节点标题。"""
    return str((source.chunk.metadata or {}).get("node_title") or "")
