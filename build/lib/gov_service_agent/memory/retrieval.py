"""长期上下文记忆混合检索。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gov_service_agent.memory.models import ContextMemoryHit, ContextMemoryItem, memory_doc_id
from gov_service_agent.memory.store import ContextMemoryStore
from gov_service_agent.rag.engine.scoring import bm25_retrieve
from gov_service_agent.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from gov_service_agent.search.keyword import KeywordIndex
from gov_service_agent.search.vector.base import VectorIndex


class ContextMemoryRetriever:
    """长期记忆语义和全文混合检索器。"""

    def __init__(
        self,
        *,
        store: ContextMemoryStore,
        vector_index: VectorIndex,
        embedding_provider: EmbeddingProvider,
        keyword_index: KeywordIndex | None = None,
        distance_threshold: float = 0.4,
        keyword_min_score: float = 0.2,
    ) -> None:
        """初始化长期记忆检索器。"""
        self.store = store
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.keyword_index = keyword_index
        self.distance_threshold = max(0.0, min(1.0, float(0.4 if distance_threshold is None else distance_threshold)))
        self.keyword_min_score = max(0.0, min(1.0, float(0.2 if keyword_min_score is None else keyword_min_score)))

    def index_memories(self, memories: list[ContextMemoryItem]) -> None:
        """写入长期记忆向量和全文索引。"""
        chunks = [memory.to_chunk() for memory in memories if memory.content.strip()]
        if not chunks:
            return
        vectors = self.embedding_provider.embed_documents([chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            self.vector_index.upsert(chunk, vector)
        flush = getattr(self.vector_index, "flush", None)
        if callable(flush):
            flush()
        if self.keyword_index is not None:
            self.keyword_index.upsert_chunks(chunks)

    def retrieve(self, query: str, *, user_id: int, limit: int) -> tuple[list[ContextMemoryHit], dict[str, Any]]:
        """按用户隔离检索长期记忆。"""
        normalized_query = " ".join(str(query or "").split())
        if not normalized_query:
            return [], self._status([], [], retrieval_mode="empty_query")
        filters = {"doc_id": memory_doc_id(user_id)}
        dense_hits = self._dense_hits(normalized_query, filters=filters, limit=max(limit * 3, limit))
        keyword_hits = self._keyword_hits(normalized_query, filters=filters, user_id=user_id, limit=max(limit * 3, limit))
        fused = self._fuse(dense_hits, keyword_hits)
        selected = fused[: max(1, int(limit or 5))]
        return selected, self._status(dense_hits, keyword_hits, retrieval_mode="hybrid")

    def _dense_hits(self, query: str, *, filters: dict[str, Any], limit: int) -> list[ContextMemoryHit]:
        """执行向量检索并应用 distance 阈值。"""
        query_vector = self.embedding_provider.embed_query(query)
        scored_ids = self.vector_index.search(query_vector, filters, limit=limit)
        accepted_ids = [
            memory_id
            for memory_id, score in scored_ids
            if (1.0 - float(score)) <= self.distance_threshold
        ]
        memories_by_id = {memory.id: memory for memory in self.store.get_memories_by_ids(accepted_ids)}
        hits = []
        for memory_id, score in scored_ids:
            memory = memories_by_id.get(memory_id)
            if not memory:
                continue
            hits.append(ContextMemoryHit(memory=memory, score=float(score), dense_score=float(score), retrieval_channel="dense"))
        return hits

    def _keyword_hits(
        self,
        query: str,
        *,
        filters: dict[str, Any],
        user_id: int,
        limit: int,
    ) -> list[ContextMemoryHit]:
        """执行全文检索；没有外部全文索引时用本地 BM25 兜底。"""
        if self.keyword_index is not None:
            scored_ids = [
                item
                for item in self.keyword_index.search(query, filters, limit=limit)
                if float(item[1]) >= self.keyword_min_score
            ]
            memories_by_id = {memory.id: memory for memory in self.store.get_memories_by_ids([item[0] for item in scored_ids])}
            return [
                ContextMemoryHit(
                    memory=memories_by_id[memory_id],
                    score=float(score),
                    bm25_score=float(score),
                    retrieval_channel="bm25",
                )
                for memory_id, score in scored_ids
                if memory_id in memories_by_id
            ]
        memories = self.store.list_user_memories(user_id=user_id, limit=200)
        local_hits = bm25_retrieve(query, [memory.to_chunk() for memory in memories], limit=limit)
        memories_by_id = {memory.id: memory for memory in memories}
        return [
            ContextMemoryHit(
                memory=memories_by_id[source.chunk.id],
                score=source.bm25_score,
                bm25_score=source.bm25_score,
                retrieval_channel="bm25",
            )
            for source in local_hits
            if source.bm25_score >= self.keyword_min_score and source.chunk.id in memories_by_id
        ]

    def _fuse(self, dense_hits: list[ContextMemoryHit], keyword_hits: list[ContextMemoryHit]) -> list[ContextMemoryHit]:
        """合并 dense 和 BM25 命中。"""
        scores: dict[str, float] = defaultdict(float)
        dense_scores: dict[str, float] = defaultdict(float)
        keyword_scores: dict[str, float] = defaultdict(float)
        memories: dict[str, ContextMemoryItem] = {}
        channels: dict[str, set[str]] = defaultdict(set)
        for hits in [dense_hits, keyword_hits]:
            for rank, hit in enumerate(hits, start=1):
                memory_id = hit.memory.id
                scores[memory_id] += 1.0 / (60 + rank)
                dense_scores[memory_id] = max(dense_scores[memory_id], hit.dense_score)
                keyword_scores[memory_id] = max(keyword_scores[memory_id], hit.bm25_score)
                memories[memory_id] = hit.memory
                channels[memory_id].add(hit.retrieval_channel)
        output = [
            ContextMemoryHit(
                memory=memories[memory_id],
                score=score + memories[memory_id].importance * 0.01,
                dense_score=dense_scores[memory_id],
                bm25_score=keyword_scores[memory_id],
                retrieval_channel="+".join(sorted(channels[memory_id])),
            )
            for memory_id, score in scores.items()
        ]
        return sorted(output, key=lambda item: item.score, reverse=True)

    def _status(self, dense_hits: list[ContextMemoryHit], keyword_hits: list[ContextMemoryHit], *, retrieval_mode: str) -> dict[str, Any]:
        """构建长期记忆检索诊断状态。"""
        return {
            "retrieval_mode": retrieval_mode,
            "distance_threshold": self.distance_threshold,
            "dense_min_similarity": round(1.0 - self.distance_threshold, 4),
            "keyword_min_score": self.keyword_min_score,
            "channels": {
                "dense": len(dense_hits),
                "bm25": len(keyword_hits),
            },
            "vector_backend": self.vector_index.backend,
            "keyword_backend": self.keyword_index.backend if self.keyword_index is not None else "local_bm25",
        }
