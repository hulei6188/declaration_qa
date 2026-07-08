"""检索查询去重、来源合并和答案来源选择。"""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import RetrievalResult, RetrievedSource
from declaration_qa.rag.engine.text import normalize_text


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


def select_answer_sources(
    merged: dict[str, RetrievedSource],
    query: str,
    *,
    limit: int = 6,
    retrieval_plan: dict[str, Any] | None = None,
) -> list[RetrievedSource]:
    """选择进入答案生成的来源。"""
    ranked = sorted(
        merged.values(),
        key=lambda source: (float(source.rerank_score or 0.0), float(source.score or 0.0)),
        reverse=True,
    )
    return ranked[:limit]
