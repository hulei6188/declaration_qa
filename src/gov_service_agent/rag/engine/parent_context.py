"""父级上下文扩展。"""

from __future__ import annotations

from dataclasses import replace

from gov_service_agent.core.domain import KnowledgeChunk, RetrievedSource


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
    parent_deduped = _dedupe_expanded_parent_sources(expanded)
    return dedupe_asset_sources(parent_deduped)


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


def dedupe_asset_sources(sources: list[RetrievedSource]) -> list[RetrievedSource]:
    """合并同一视觉资产的原图、OCR 和视觉摘要命中。"""
    deduped: list[RetrievedSource] = []
    by_key: dict[str, int] = {}
    for source in sources:
        asset_id = _source_asset_id(source)
        if not asset_id:
            deduped.append(source)
            continue
        key = f"asset:{source.chunk.doc_id}:{asset_id}"
        if key not in by_key:
            by_key[key] = len(deduped)
            deduped.append(source)
            continue
        existing_index = by_key[key]
        deduped[existing_index] = _merge_asset_sources(deduped[existing_index], source, asset_id=asset_id)
    return deduped


def _source_asset_id(source: RetrievedSource) -> str:
    """返回原图及其派生文本共享的资产标识。"""
    metadata = source.chunk.metadata or {}
    return str(metadata.get("source_asset_id") or metadata.get("asset_id") or "").strip()


def _merge_asset_sources(left: RetrievedSource, right: RetrievedSource, *, asset_id: str) -> RetrievedSource:
    """优先保留原图，并汇总同资产命中的分数、通道和片段。"""
    winner = _preferred_asset_source(left, right)
    chunk_ids = _merged_asset_chunk_ids(left, right)
    semantic_types = _merged_asset_semantic_types(left, right)
    merged_metadata = {
        **winner.chunk.metadata,
        "asset_group_id": asset_id,
        "asset_merged_chunk_ids": chunk_ids,
        "asset_merged_chunk_count": len(chunk_ids),
        "asset_merged_semantic_types": semantic_types,
    }
    return replace(
        winner,
        chunk=replace(winner.chunk, metadata=merged_metadata),
        score=max(float(left.score or 0.0), float(right.score or 0.0)),
        dense_score=max(float(left.dense_score or 0.0), float(right.dense_score or 0.0)),
        bm25_score=max(float(left.bm25_score or 0.0), float(right.bm25_score or 0.0)),
        rerank_score=max(float(left.rerank_score or 0.0), float(right.rerank_score or 0.0)),
        retrieval_channel=_merge_channels(left.retrieval_channel, right.retrieval_channel),
    )


def _preferred_asset_source(left: RetrievedSource, right: RetrievedSource) -> RetrievedSource:
    """同资产命中优先使用可展示、可供视觉模型读取的原图。"""
    left_is_original = _is_original_image_source(left)
    right_is_original = _is_original_image_source(right)
    if left_is_original != right_is_original:
        return left if left_is_original else right
    return left if _source_rank(left) >= _source_rank(right) else right


def _is_original_image_source(source: RetrievedSource) -> bool:
    """判断来源是否为资产原图而非 OCR/摘要派生块。"""
    metadata = source.chunk.metadata or {}
    return bool(metadata.get("asset_id")) and not bool(metadata.get("source_asset_id")) and metadata.get("modality") == "image"


def _merged_asset_chunk_ids(*sources: RetrievedSource) -> list[str]:
    """合并同一资产已经命中的 chunk ID。"""
    values: list[str] = []
    seen: set[str] = set()
    for source in sources:
        metadata = source.chunk.metadata or {}
        chunk_ids = metadata.get("asset_merged_chunk_ids") or [source.chunk.id]
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        for value in chunk_ids:
            chunk_id = str(value or "").strip()
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                values.append(chunk_id)
    return values


def _merged_asset_semantic_types(*sources: RetrievedSource) -> list[str]:
    """记录资产组中实际命中的语义块类型。"""
    values: list[str] = []
    seen: set[str] = set()
    for source in sources:
        metadata = source.chunk.metadata or {}
        semantic_types = metadata.get("asset_merged_semantic_types") or [metadata.get("chunk_type")]
        if isinstance(semantic_types, str):
            semantic_types = [semantic_types]
        for value in semantic_types:
            semantic_type = str(value or "").strip()
            if semantic_type and semantic_type not in seen:
                seen.add(semantic_type)
                values.append(semantic_type)
    return values


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
