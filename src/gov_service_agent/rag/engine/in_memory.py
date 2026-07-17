"""内存 RAG 引擎实现。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from gov_service_agent.rag.engine.base import BaseRAGEngine, strategy_weighted_hit_lists
from gov_service_agent.rag.engine.embeddings import embed_knowledge_chunk_batch
from gov_service_agent.rag.engine.scoring import bm25_retrieve, dense_retrieve, matches_filter, rrf_fuse
from gov_service_agent.rag.engine.parent_context import expand_parent_context
from gov_service_agent.rag.engine.text import unique_ordered_strings
from gov_service_agent.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from gov_service_agent.rag.model_providers.rerankers.reranker import Reranker
from gov_service_agent.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage, published_visibility_filter
from gov_service_agent.search.keyword import KeywordIndex
from gov_service_agent.search.vector.base import VectorIndex
from gov_service_agent.search.vector.memory import MemoryVectorIndex
from gov_service_agent.storage.cache import CacheStore


class InMemoryRAGEngine(BaseRAGEngine):
    """基于进程内文档、片段和向量的 RAG 引擎。"""

    def __init__(
        self,
        *,
        cache_store: CacheStore | None = None,
        vector_index: VectorIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        keyword_index: KeywordIndex | None = None,
        cache_ttl_seconds: int = 600,
        rerank_score_threshold: float = 0.0,
        rerank_max_sources: int = 20,
        retrieval_candidates_per_query: int = 30,
        rerank_top_n_per_query: int = 12,
    ) -> None:
        """初始化内存 RAG 引擎。"""
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        super().__init__(
            cache_store=cache_store,
            vector_index=vector_index or MemoryVectorIndex(),
            embedding_provider=embedding_provider,
            reranker=reranker,
            keyword_index=keyword_index,
            cache_ttl_seconds=cache_ttl_seconds,
            rerank_score_threshold=rerank_score_threshold,
            rerank_max_sources=rerank_max_sources,
            retrieval_candidates_per_query=retrieval_candidates_per_query,
            rerank_top_n_per_query=rerank_top_n_per_query,
        )

    @property
    def chunk_count(self) -> int:
        """返回已加载片段数量。"""
        return len(self._chunks)

    def documents(self) -> list[KnowledgeDocument]:
        """返回已加载文档。"""
        return list(self._documents.values())

    def index_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """将片段写入本地内存和本地索引。"""
        self.delete_document(document.id)
        self._documents[document.id] = document
        child_chunks = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        vectors = embed_knowledge_chunk_batch(self.embedding_provider, child_chunks)
        vectors_by_chunk_id = {chunk.id: vector for chunk, vector in zip(child_chunks, vectors)}
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            if chunk.chunk_type == "child":
                vector = vectors_by_chunk_id[chunk.id]
                self._vectors[chunk.embedding_id] = vector
                self.vector_index.upsert(chunk, vector)
        if self.keyword_index:
            self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self._clear_cache()
        return len(chunks)

    def promote_indexed_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """复用暂存阶段的内存向量更新发布状态。"""
        self._documents[document.id] = document
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            if chunk.chunk_type == "child":
                vector = self._vectors.get(chunk.embedding_id)
                if vector is None:
                    raise RuntimeError("Staged vector is unavailable for knowledge revision promotion")
                self.vector_index.upsert(chunk, vector)
        if self.keyword_index:
            self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self._clear_cache()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """从内存和索引中删除单个文档。"""
        self._documents.pop(doc_id, None)
        for chunk_id, chunk in list(self._chunks.items()):
            if chunk.doc_id == doc_id:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk.embedding_id, None)
        self.vector_index.delete_by_doc(doc_id)
        if self.keyword_index:
            self.keyword_index.delete_by_doc(doc_id)
        self._clear_cache()

    def rebuild_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """根据片段重建本地内存、向量和关键词索引。"""
        self._documents.clear()
        self._chunks.clear()
        self._vectors.clear()
        self.vector_index.clear()
        if self.keyword_index:
            self.keyword_index.clear()
        self._documents = {document.id: document for document in documents}
        child_chunks = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        vectors = embed_knowledge_chunk_batch(self.embedding_provider, child_chunks)
        vectors_by_chunk_id = {chunk.id: vector for chunk, vector in zip(child_chunks, vectors)}
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            if chunk.chunk_type == "child":
                vector = vectors_by_chunk_id[chunk.id]
                self._vectors[chunk.embedding_id] = vector
                self.vector_index.upsert(chunk, vector)
        if self.keyword_index:
            self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self.cache_store.clear()
        return len(chunks)

    def load_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """加载持久化文档和片段元数据，不重新向量化。"""
        self._documents = {document.id: document for document in documents}
        self._chunks = {chunk.id: chunk for chunk in chunks}
        self._vectors.clear()
        self.cache_store.clear()
        return len(chunks)

    def _retrieve_uncached(
        self,
        normalized_query: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """从内存片段中检索。"""
        search_filter = published_visibility_filter(effective_filter)
        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, search_filter)]
        prefilter_child_count = self._prefilter_child_count()
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        if not candidates:
            return self._empty_result(
                normalized_query,
                effective_filter,
                top_k=top_k,
                prefilter_child_count=prefilter_child_count,
                coverage=coverage,
            )
        limit = self._recall_limit(top_k)
        degraded: list[dict[str, str]] = []
        try:
            dense_hits = self._dense_retrieve(normalized_query, candidates, search_filter, limit=limit)
        except Exception as exc:
            dense_hits = []
            degraded.append({"stage": "dense_retrieve", "error": type(exc).__name__})
        bm25_hits = bm25_retrieve(normalized_query, candidates, limit=limit)
        return self._build_result_from_hits(
            normalized_query,
            effective_filter,
            candidates,
            prefilter_child_count,
            coverage,
            dense_hits,
            bm25_hits,
            top_k=top_k,
            degraded=degraded,
            retrieval_plan=retrieval_plan,
        )

    def _retrieve_many_uncached(
        self,
        query_specs: list[dict[str, Any]],
        results: list[RetrievalResult | None],
        effective_filter: dict[str, Any],
        *,
        retrieval_plan: dict[str, Any],
    ) -> list[RetrievalResult]:
        """从内存片段中批量检索。"""
        search_filter = published_visibility_filter(effective_filter)
        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, search_filter)]
        prefilter_child_count = self._prefilter_child_count()
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        if not candidates:
            for spec in query_specs:
                result = self._empty_result(
                    str(spec["query"]),
                    effective_filter,
                    top_k=int(spec["top_k"]),
                    prefilter_child_count=prefilter_child_count,
                    coverage=coverage,
                )
                self._cache_result(str(spec["cache_key"]), result)
                results[int(spec["index"])] = result
            return [result for result in results if result is not None]

        limits = [self._recall_limit(int(spec["top_k"])) for spec in query_specs]
        dense_by_index: dict[int, list[RetrievedSource]] = {}
        degraded_by_index: dict[int, list[dict[str, str]]] = {int(spec["index"]): [] for spec in query_specs}
        try:
            query_vectors = self.embedding_provider.embed_documents([str(spec["query"]) for spec in query_specs])
            batch_hits = self.vector_index.search_batch(query_vectors, search_filter, limit=max(limits))
            by_id = {chunk.id: chunk for chunk in candidates}
            for spec, vector_hits, limit in zip(query_specs, batch_hits, limits):
                dense_by_index[int(spec["index"])] = [
                    RetrievedSource(chunk=by_id[chunk_id], score=score, dense_score=score, retrieval_channel="dense_batch")
                    for chunk_id, score in vector_hits[:limit]
                    if chunk_id in by_id and score > 0
                ]
        except Exception as exc:
            for spec in query_specs:
                degraded_by_index[int(spec["index"])].append({"stage": "dense_retrieve_batch", "error": type(exc).__name__})

        for spec, limit in zip(query_specs, limits):
            index = int(spec["index"])
            query = str(spec["query"])
            dense_hits = dense_by_index.get(index) or []
            if not dense_hits and self._vectors:
                dense_hits = dense_retrieve(query, candidates, self._vectors, limit=limit, embedding_provider=self.embedding_provider)
            bm25_hits = bm25_retrieve(query, candidates, limit=limit)
            result = self._build_result_from_hits(
                query,
                effective_filter,
                candidates,
                prefilter_child_count,
                coverage,
                dense_hits,
                bm25_hits,
                top_k=int(spec["top_k"]),
                degraded=degraded_by_index.get(index) or [],
                batch_mode=True,
                retrieval_plan=spec.get("retrieval_plan") or retrieval_plan,
            )
            self._cache_result(str(spec["cache_key"]), result)
            results[index] = result
        return [result for result in results if result is not None]

    def _retrieve_rewrites_uncached(
        self,
        original_query: str,
        query_specs: list[dict[str, Any]],
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """融合内存改写召回列表并统一重排一次。"""
        search_filter = published_visibility_filter(effective_filter)
        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, search_filter)]
        prefilter_child_count = self._prefilter_child_count()
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        query_statuses = [
            {
                "id": spec.get("id") or "",
                "strategy": spec.get("strategy") or "",
                "aspect": spec.get("aspect") or "general",
                "query": spec.get("query") or "",
                "top_k": int(spec.get("top_k") or 3),
                "source_count": 0,
                "channels": {"dense": 0, "bm25": 0},
            }
            for spec in query_specs
        ]
        if not candidates:
            return self._empty_rewrite_result(
                original_query,
                query_statuses,
                effective_filter,
                top_k=top_k,
                coverage=coverage,
                prefilter_child_count=prefilter_child_count,
            )

        limits = [self._recall_limit(int(spec["top_k"])) for spec in query_specs]
        dense_by_index: dict[int, list[RetrievedSource]] = {}
        degraded: list[dict[str, str]] = []
        try:
            query_vectors = self.embedding_provider.embed_documents([str(spec["query"]) for spec in query_specs])
            batch_hits = self.vector_index.search_batch(query_vectors, search_filter, limit=max(limits))
            by_id = {chunk.id: chunk for chunk in candidates}
            for spec, vector_hits, limit in zip(query_specs, batch_hits, limits):
                dense_by_index[int(spec["index"])] = [
                    RetrievedSource(
                        chunk=by_id[chunk_id],
                        score=score,
                        dense_score=score,
                        retrieval_channel=f"dense_batch:q{int(spec['index'])}",
                    )
                    for chunk_id, score in vector_hits[:limit]
                    if chunk_id in by_id and score > 0
                ]
        except Exception as exc:
            degraded.append({"stage": "dense_retrieve_batch", "error": type(exc).__name__})

        hit_lists: list[list[RetrievedSource]] = []
        for status, spec, limit in zip(query_statuses, query_specs, limits):
            index = int(spec["index"])
            query = str(spec["query"])
            dense_hits = dense_by_index.get(index) or []
            if not dense_hits and self._vectors:
                dense_hits = dense_retrieve(query, candidates, self._vectors, limit=limit, embedding_provider=self.embedding_provider)
            bm25_hits = bm25_retrieve(query, candidates, limit=limit)
            weighted_lists, channel_weights = strategy_weighted_hit_lists(
                str(spec.get("strategy") or "hybrid"),
                dense_hits,
                bm25_hits,
            )
            query_hits = rrf_fuse(weighted_lists, rrf_k=60) if weighted_lists else []
            if query_hits:
                hit_lists.append(query_hits)
            unique_chunk_ids = {source.chunk.id for source in [*dense_hits, *bm25_hits]}
            status["source_count"] = len(unique_chunk_ids)
            status["channels"] = {"dense": len(dense_hits), "bm25": len(bm25_hits)}
            status["channel_weights"] = channel_weights
            status["coverage_candidate"] = query_hits[0].chunk.id if query_hits else ""
        return self._build_rewrite_result_from_hit_lists(
            original_query,
            effective_filter,
            query_specs,
            query_statuses,
            hit_lists,
            coverage,
            top_k=top_k,
            degraded=degraded,
            candidate_count=len(candidates),
            prefilter_child_count=prefilter_child_count,
            retrieval_plan=retrieval_plan,
        )

    def _retrieve_image_uncached(
        self,
        image_path: str,
        text: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """使用用户上传图片向量检索内存索引。"""
        search_filter = published_visibility_filter(effective_filter)
        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, search_filter)]
        prefilter_child_count = self._prefilter_child_count()
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        result_query = self.image_rerank_query(text, retrieval_plan)
        if not candidates:
            return self._empty_result(
                result_query,
                effective_filter,
                top_k=top_k,
                prefilter_child_count=prefilter_child_count,
                coverage=coverage,
            )
        limit = self._recall_limit(top_k)
        degraded: list[dict[str, str]] = []
        try:
            query_vector = self.embed_image_query(image_path, text)
            by_id = {chunk.id: chunk for chunk in candidates}
            vector_hits = self.vector_index.search(query_vector, search_filter, limit=limit)
            dense_hits = [
                RetrievedSource(chunk=by_id[chunk_id], score=score, dense_score=score, retrieval_channel="image_dense")
                for chunk_id, score in vector_hits[:limit]
                if chunk_id in by_id and score > 0
            ]
        except Exception as exc:
            dense_hits = []
            degraded.append({"stage": "image_dense_retrieve", "error": type(exc).__name__})
        return self._build_result_from_hits(
            result_query,
            effective_filter,
            candidates,
            prefilter_child_count,
            coverage,
            dense_hits,
            [],
            top_k=top_k,
            degraded=degraded,
            candidate_source="image_query",
            retrieval_plan=retrieval_plan,
        )

    def _prefilter_child_count(self) -> int:
        """返回已加载子片段数量。"""
        return sum(1 for chunk in self._chunks.values() if chunk.chunk_type == "child")

    def _load_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """从内存加载片段。"""
        unique_ids = unique_ordered_strings(chunk_ids)
        chunks = [self._chunks[chunk_id] for chunk_id in unique_ids if chunk_id in self._chunks]
        by_id = {chunk.id: chunk for chunk in chunks}
        return [by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id]

    def _expand_parent_context(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        """用内存中的父片段扩展子命中。"""
        return expand_parent_context(sources, self._chunks)

    def _dense_retrieve(
        self,
        query: str,
        candidates: list[KnowledgeChunk],
        filters: dict[str, Any],
        *,
        limit: int,
    ) -> list[RetrievedSource]:
        """为当前引擎执行向量检索。"""
        query_vector = self.embedding_provider.embed_query(query)
        by_id = {chunk.id: chunk for chunk in candidates}
        vector_hits = self.vector_index.search(query_vector, filters, limit=limit)
        hits = [
            RetrievedSource(chunk=by_id[chunk_id], score=score, dense_score=score, retrieval_channel="dense")
            for chunk_id, score in vector_hits
            if chunk_id in by_id and score > 0
        ]
        if hits:
            return hits[:limit]
        if not self._vectors:
            return []
        return dense_retrieve(query, candidates, self._vectors, limit=limit, embedding_provider=self.embedding_provider)
