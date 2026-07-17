"""检索查询去重、来源合并和答案来源选择。"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from gov_service_agent.core.domain import RetrievalResult, RetrievedSource
from gov_service_agent.rag.engine.parent_context import dedupe_asset_sources
from gov_service_agent.rag.engine.text import normalize_text


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
        "aspect": retrieval_query.get("aspect") or "general",
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
    limit: int | None = 6,
    retrieval_plan: dict[str, Any] | None = None,
) -> list[RetrievedSource]:
    """选择进入答案生成的来源；limit=None 时仅去重，不再截断。"""
    ranked = sorted(
        merged.values(),
        key=lambda source: (float(source.rerank_score or 0.0), float(source.score or 0.0)),
        reverse=True,
    )
    deduped = dedupe_asset_sources(ranked)
    if limit is None:
        return deduped
    return deduped[: max(0, int(limit))]


def merge_query_results_with_coverage(
    queries: list[dict[str, Any]],
    results: list[RetrievalResult],
    *,
    search_round: int,
    min_per_query: int,
    max_sources: int,
) -> tuple[list[RetrievedSource], dict[str, Any]]:
    """按查询最低配额和局部名次轮询合并已独立重排的结果。"""
    entries: list[dict[str, Any]] = []
    aliases: dict[str, int] = {}
    group_keys: dict[str, list[int]] = {}
    group_order: list[str] = []
    for query, result in zip(queries, results):
        query_id = str(query.get("id") or f"q{len(group_order) + 1}")
        match_id = f"r{search_round}:{query_id}"
        group_order.append(match_id)
        group_keys[match_id] = []
        for local_rank, source in enumerate(result.sources, start=1):
            entry_index = _retrieved_source_entry(entries, aliases, source)
            match = {
                "match_id": match_id,
                "search_round": search_round,
                "query_id": query_id,
                "query": str(query.get("text") or ""),
                "strategy": str(query.get("strategy") or "hybrid"),
                "local_rank": local_rank,
            }
            if match_id not in {item["match_id"] for item in entries[entry_index]["matches"]}:
                entries[entry_index]["matches"].append(match)
            if entry_index not in group_keys[match_id]:
                group_keys[match_id].append(entry_index)

    selected_indexes = _coverage_selection(
        group_order,
        group_keys,
        min_per_group=min_per_query,
        max_sources=max_sources,
    )
    selected = [
        _annotate_retrieved_source(entries[index]["source"], entries[index]["matches"])
        for index in selected_indexes
    ]
    groups = []
    selected_set = set(selected_indexes)
    for match_id in group_order:
        available = len(group_keys[match_id])
        selected_count = sum(1 for index in group_keys[match_id] if index in selected_set)
        groups.append(
            {
                "match_id": match_id,
                "available": available,
                "selected": selected_count,
                "requested_minimum": min_per_query,
                "minimum_met": selected_count >= min(min_per_query, available),
            }
        )
    return selected, {
        "mode": "per_query_rerank_coverage_merge",
        "query_count": len(group_order),
        "deduped_candidate_count": len(entries),
        "selected_count": len(selected),
        "max_sources": max_sources,
        "min_sources_per_query": min_per_query,
        "queries": groups,
    }


def merge_accumulated_source_payloads(
    sources: list[dict[str, Any]],
    *,
    min_per_query: int,
    max_sources: int,
) -> list[dict[str, Any]]:
    """跨 Agent 检索轮次去重并保留各轮各查询的最低覆盖。"""
    entries: list[dict[str, Any]] = []
    aliases: dict[str, int] = {}
    group_keys: dict[str, list[int]] = {}
    group_order: list[str] = []
    group_rounds: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        entry_index = _payload_source_entry(entries, aliases, source)
        matches = _payload_matches(source)
        existing_match_ids = {item["match_id"] for item in entries[entry_index]["matches"]}
        for match in matches:
            match_id = match["match_id"]
            if match_id not in group_keys:
                group_order.append(match_id)
                group_keys[match_id] = []
                group_rounds[match_id] = int(match.get("search_round") or 0)
            if match_id not in existing_match_ids:
                entries[entry_index]["matches"].append(match)
                existing_match_ids.add(match_id)
            if entry_index not in group_keys[match_id]:
                group_keys[match_id].append(entry_index)

    required_slots = sum(
        min(min_per_query, len(group_keys.get(group, [])))
        for group in group_order
    )
    coverage_order = (
        sorted(group_order, key=lambda item: group_rounds.get(item, 0), reverse=True)
        if required_slots > max_sources
        else group_order
    )
    selected_indexes = _coverage_selection(
        coverage_order,
        group_keys,
        min_per_group=min_per_query,
        max_sources=max_sources,
    )
    output = []
    for index in selected_indexes:
        source = dict(entries[index]["source"])
        metadata = dict(source.get("metadata") or {})
        matches = entries[index]["matches"]
        metadata.update(_match_metadata(matches))
        source["metadata"] = metadata
        output.append(source)
    return output


def _coverage_selection(
    group_order: list[str],
    group_keys: dict[str, list[int]],
    *,
    min_per_group: int,
    max_sources: int,
) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(index: int) -> None:
        if index not in selected_set and len(selected) < max_sources:
            selected.append(index)
            selected_set.add(index)

    for group in group_order:
        for index in group_keys.get(group, [])[:min_per_group]:
            add(index)
    max_depth = max((len(group_keys.get(group, [])) for group in group_order), default=0)
    for depth in range(max_depth):
        for group in group_order:
            keys = group_keys.get(group, [])
            if depth < len(keys):
                add(keys[depth])
            if len(selected) >= max_sources:
                return selected
    return selected


def _retrieved_source_entry(
    entries: list[dict[str, Any]],
    aliases: dict[str, int],
    source: RetrievedSource,
) -> int:
    keys = _source_keys(source.chunk.id, source.chunk.doc_id, source.chunk.content, source.chunk.metadata)
    existing = next((aliases[key] for key in keys if key in aliases), None)
    if existing is None:
        existing = len(entries)
        entries.append({"source": source, "matches": []})
    elif _source_score(source) > _source_score(entries[existing]["source"]):
        entries[existing]["source"] = source
    for key in keys:
        aliases[key] = existing
    return existing


def _payload_source_entry(
    entries: list[dict[str, Any]],
    aliases: dict[str, int],
    source: dict[str, Any],
) -> int:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    keys = _source_keys(
        str(source.get("chunk_id") or ""),
        str(source.get("doc_id") or ""),
        str(source.get("evidence_text") or source.get("content") or source.get("snippet") or ""),
        metadata,
    )
    existing = next((aliases[key] for key in keys if key in aliases), None)
    if existing is None:
        existing = len(entries)
        entries.append({"source": source, "matches": []})
    elif _payload_score(source) > _payload_score(entries[existing]["source"]):
        entries[existing]["source"] = source
    for key in keys:
        aliases[key] = existing
    return existing


def _source_keys(chunk_id: str, doc_id: str, content: str, metadata: dict[str, Any]) -> list[str]:
    keys = [f"chunk:{chunk_id}"] if chunk_id else []
    text_key = re.sub(r"\s+", "", content).lower()
    if text_key:
        keys.append(f"text:{text_key}")
    chunk_type = str(metadata.get("chunk_type") or metadata.get("semantic_type") or "")
    node_title = str(metadata.get("node_title") or "").strip()
    if chunk_type == "table_kv" and node_title:
        keys.append(f"table:{doc_id}|{metadata.get('section_title') or ''}|{node_title}")
    return keys or [f"anonymous:{id(metadata)}"]


def _annotate_retrieved_source(source: RetrievedSource, matches: list[dict[str, Any]]) -> RetrievedSource:
    metadata = {**(source.chunk.metadata or {}), **_match_metadata(matches)}
    return replace(source, chunk=replace(source.chunk, metadata=metadata))


def _match_metadata(matches: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(matches, key=lambda item: (int(item.get("search_round") or 0), int(item.get("local_rank") or 0)))
    return {
        "retrieval_matches": ordered,
        "matched_query_ids": [item["match_id"] for item in ordered],
        "best_local_rank": min((int(item.get("local_rank") or 0) for item in ordered), default=0),
        "search_round": min((int(item.get("search_round") or 0) for item in ordered), default=0),
    }


def _payload_matches(source: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    matches = [dict(item) for item in metadata.get("retrieval_matches") or [] if isinstance(item, dict)]
    if matches:
        return matches
    search_round = int(metadata.get("search_round") or 0)
    match_id = f"r{search_round}:legacy"
    return [{"match_id": match_id, "search_round": search_round, "query_id": "legacy", "local_rank": 1}]


def _source_score(source: RetrievedSource) -> tuple[float, float]:
    return float(source.rerank_score or 0.0), float(source.score or 0.0)


def _payload_score(source: dict[str, Any]) -> tuple[float, float]:
    return float(source.get("rerank_score") or 0.0), float(source.get("score") or 0.0)
