"""父级上下文扩展。"""

from __future__ import annotations

from dataclasses import replace

from declaration_qa.core.domain import KnowledgeChunk, RetrievedSource


def expand_parent_context(sources: list[RetrievedSource], chunks: dict[str, KnowledgeChunk]) -> list[RetrievedSource]:
    """使用父片段内容扩展子命中。"""
    expanded = []
    for source in sources:
        chunk = source.chunk
        parent = chunks.get(chunk.parent_id) if chunk.chunk_type == "child" else None
        if parent and parent.content and parent.content != chunk.content:
            expanded_chunk = replace(
                chunk,
                content=parent.content,
                metadata={
                    **chunk.metadata,
                    "retrieved_child_chunk_id": chunk.id,
                    "expanded_parent_chunk_id": parent.id,
                },
            )
            expanded.append(replace(source, chunk=expanded_chunk))
        else:
            expanded.append(source)
    return _dedupe_expanded_parent_sources(expanded)


def _dedupe_expanded_parent_sources(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    """合并展开到同一父片段的多个子命中。"""
    deduped: list[RetrievedSource] = []
    by_key: dict[str, int] = {}
    for source in sources:
        key = _parent_context_dedupe_key(source)
        if key not in by_key:
            by_key[key] = len(deduped)
            deduped.append(source)
            continue
        existing_index = by_key[key]
        merged = _merge_parent_context_sources(deduped[existing_index], source)
        deduped[existing_index] = merged
    return deduped


def _parent_context_dedupe_key(source: RetrievedSource) -> str:
    """返回父级上下文去重键。"""
    metadata = source.chunk.metadata or {}
    parent_id = str(metadata.get("expanded_parent_chunk_id") or "").strip()
    if parent_id:
        return f"parent:{source.chunk.doc_id}:{parent_id}"
    return f"chunk:{source.chunk.doc_id}:{source.chunk.id}"


def _merge_parent_context_sources(left: RetrievedSource, right: RetrievedSource) -> RetrievedSource:
    """保留更高分来源，并记录所有命中的子片段。"""
    winner = left if _source_rank(left) >= _source_rank(right) else right
    child_ids = _merged_child_ids(left, right)
    merged_metadata = {
        **winner.chunk.metadata,
        "retrieved_child_chunk_ids": child_ids,
        "parent_context_merged_child_count": len(child_ids),
    }
    merged_channel = _merge_channels(left.retrieval_channel, right.retrieval_channel)
    return replace(
        winner,
        chunk=replace(winner.chunk, metadata=merged_metadata),
        retrieval_channel=merged_channel,
    )


def _source_rank(source: RetrievedSource) -> tuple[float, float, float, float]:
    """按重排、综合、向量、关键词分数选择展示来源。"""
    return (
        float(source.rerank_score or 0.0),
        float(source.score or 0.0),
        float(source.dense_score or 0.0),
        float(source.bm25_score or 0.0),
    )


def _merged_child_ids(*sources: RetrievedSource) -> list[str]:
    """合并父级展开前命中的子片段 ID。"""
    child_ids: list[str] = []
    seen: set[str] = set()
    for source in sources:
        metadata = source.chunk.metadata or {}
        values = metadata.get("retrieved_child_chunk_ids") or [metadata.get("retrieved_child_chunk_id") or source.chunk.id]
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            values = [source.chunk.id]
        for value in values:
            child_id = str(value or "").strip()
            if child_id and child_id not in seen:
                seen.add(child_id)
                child_ids.append(child_id)
    return child_ids


def _merge_channels(left: str, right: str) -> str:
    """合并检索通道标签。"""
    channels: list[str] = []
    seen: set[str] = set()
    for value in (left, right):
        for channel in str(value or "").split("+"):
            channel = channel.strip()
            if channel and channel not in seen:
                seen.add(channel)
                channels.append(channel)
    return "+".join(channels) or "hybrid"
