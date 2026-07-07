"""检索评分和元数据过滤。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, RetrievedSource
from declaration_qa.rag.engine.text import normalize_text, tokenize
from declaration_qa.rag.engine.vectors import cosine


def dense_retrieve(
    query: str,
    candidates: list[KnowledgeChunk],
    vectors: dict[str, list[float]],
    *,
    limit: int,
    embedding_provider: Any,
) -> list[RetrievedSource]:
    """基于内存向量执行本地稠密检索。"""
    query_vector = embedding_provider.embed_query(query)
    hits = []
    for chunk in candidates:
        vector = vectors.get(chunk.embedding_id)
        if not vector:
            continue
        score = cosine(query_vector, vector)
        if score > 0:
            hits.append(RetrievedSource(chunk=chunk, score=score, dense_score=score, retrieval_channel="dense"))
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def bm25_retrieve(query: str, candidates: list[KnowledgeChunk], *, limit: int) -> list[RetrievedSource]:
    """执行本地 BM25 关键词检索。"""
    query_tokens = tokenize(query)
    corpus = [tokenize(chunk.content) for chunk in candidates]
    if not query_tokens:
        return []
    scores = bm25_scores(query_tokens, corpus)
    max_score = max(scores or [0.0]) or 1.0
    hits = []
    for chunk, score in zip(candidates, scores):
        normalized_score = float(score) / max_score if score > 0 else 0.0
        if normalized_score <= 0:
            continue
        hits.append(RetrievedSource(chunk=chunk, score=normalized_score, bm25_score=normalized_score, retrieval_channel="bm25"))
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def bm25_scores(query_tokens: list[str], corpus: list[list[str]]) -> list[float]:
    """计算 BM25 分数，并使用词法兜底。"""
    try:
        from rank_bm25 import BM25Okapi

        scores = [float(score) for score in BM25Okapi(corpus).get_scores(query_tokens)]
        if any(score > 0 for score in scores):
            return scores
    except Exception:
        pass
    query_set = set(query_tokens)
    return [float(len(query_set.intersection(tokens))) for tokens in corpus]


def rrf_fuse(hit_lists: list[list[RetrievedSource]], *, rrf_k: int) -> list[RetrievedSource]:
    """使用倒数排名融合合并检索通道。"""
    chunk_scores: dict[str, float] = defaultdict(float)
    best_source: dict[str, RetrievedSource] = {}
    dense_scores: dict[str, float] = defaultdict(float)
    bm25_scores_by_chunk: dict[str, float] = defaultdict(float)
    channels: dict[str, set[str]] = defaultdict(set)
    for hit_list in hit_lists:
        for rank, hit in enumerate(hit_list, start=1):
            chunk_id = hit.chunk.id
            chunk_scores[chunk_id] += 1.0 / (rrf_k + rank)
            best_source.setdefault(chunk_id, hit)
            dense_scores[chunk_id] = max(dense_scores[chunk_id], hit.dense_score)
            bm25_scores_by_chunk[chunk_id] = max(bm25_scores_by_chunk[chunk_id], hit.bm25_score)
            channels[chunk_id].add(hit.retrieval_channel)
    fused = []
    for chunk_id, score in chunk_scores.items():
        source = best_source[chunk_id]
        fused.append(
            RetrievedSource(
                chunk=source.chunk,
                score=score,
                dense_score=dense_scores[chunk_id],
                bm25_score=bm25_scores_by_chunk[chunk_id],
                retrieval_channel="+".join(sorted(channels[chunk_id])),
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)


def metadata_match_boost(chunk: KnowledgeChunk, filters: dict[str, Any]) -> float:
    """元数据匹配过滤条件时提升重排分数。"""
    if not filters:
        return 0.0
    metadata = chunk.metadata or {}
    matched = sum(
        1
        for key, value in filters.items()
        if key not in {"as_of", "rollout_key"} and metadata_value_matches(chunk_filter_value(chunk, metadata, key), value)
    )
    return min(0.2, matched * 0.05)


def matches_filter(chunk: KnowledgeChunk, filters: dict[str, Any]) -> bool:
    """返回片段是否满足元数据过滤条件。"""
    if not filters:
        return True
    metadata = chunk.metadata or {}
    for key, value in filters.items():
        if key in {"as_of", "rollout_key"}:
            continue
        if not metadata_value_matches(chunk_filter_value(chunk, metadata, key), value):
            return False
    return True


def chunk_filter_value(chunk: KnowledgeChunk, metadata: dict[str, Any], key: str) -> Any:
    """返回用于元数据过滤匹配的值。"""
    if key == "status":
        return chunk.status
    if key == "doc_status":
        return metadata.get("doc_status") or chunk.status
    if key == "audit_node":
        return chunk.audit_node or metadata.get("audit_node")
    return metadata.get(key)


def metadata_value_matches(candidate: Any, expected: Any) -> bool:
    """匹配元数据值，并支持列表取值。"""
    if isinstance(expected, list):
        return any(metadata_value_matches(candidate, item) for item in expected)
    if isinstance(candidate, list):
        return any(metadata_value_matches(item, expected) for item in candidate)
    return normalize_text(candidate) == normalize_text(expected)
