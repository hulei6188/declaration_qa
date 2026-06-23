"""知识文档切分、索引、混合检索和种子数据构建。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import replace
from datetime import date
from typing import Any, Iterable, Protocol

from declaration_qa.cache import CacheStore, InMemoryCacheStore
from declaration_qa.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.keyword_index import KeywordIndex
from declaration_qa.model_providers import EmbeddingProvider, HashEmbeddingProvider, HeuristicReranker, Reranker
from declaration_qa.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage, normalize_retrieval_filter, sanitize_metadata
from declaration_qa.vector_index import MemoryVectorIndex, VectorIndex


EMBEDDING_DIMS = 96
CHUNKING_PARENT_CHILD = "parent_child"
CHUNKING_HIERARCHICAL = "hierarchical"
SUPPORTED_CHUNKING_STRATEGIES = {CHUNKING_PARENT_CHILD, CHUNKING_HIERARCHICAL}


class ChunkStore(Protocol):
    """Persistent chunk store used for on-demand retrieval hydration."""

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """Load chunks by public chunk ids."""
        ...

    def get_parent_chunks(self, parent_ids: list[str]) -> list[KnowledgeChunk]:
        """Load parent chunks by public chunk ids."""
        ...

    def get_structured_section_chunks(self, parent_section_ids: list[str], target_chunk_type: str) -> list[KnowledgeChunk]:
        """Load structured sibling chunks for section backfill."""
        ...


class InMemoryRAGEngine:
    """Isolated RAG engine for the declaration QA workflow.

    The class keeps the same retrieval shape expected in production
    (dense retrieval -> BM25 -> RRF -> rerank), while retaining metadata only
    for source display and auditing. It uses an in-memory vector index so it can
    run without Milvus/Redis locally.
    """

    def __init__(
        self,
        *,
        cache_store: CacheStore | None = None,
        vector_index: VectorIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        keyword_index: KeywordIndex | None = None,
        chunk_store: ChunkStore | None = None,
        load_chunks_into_memory: bool = True,
        cache_ttl_seconds: int = 600,
    ) -> None:
        """初始化InMemoryRAGEngine。"""
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self.cache_store = cache_store or InMemoryCacheStore()
        self.vector_index = vector_index or MemoryVectorIndex()
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=EMBEDDING_DIMS)
        self.reranker = reranker or HeuristicReranker()
        self.keyword_index = keyword_index
        self.chunk_store = chunk_store
        self.load_chunks_into_memory = bool(load_chunks_into_memory)
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))

    @property
    def chunk_count(self) -> int:
        """返回当前索引中的片段数量。"""
        return len(self._chunks)

    def documents(self) -> list[KnowledgeDocument]:
        """返回当前知识文档列表。"""
        return list(self._documents.values())

    def index_document(self, document: KnowledgeDocument) -> int:
        """索引单篇知识文档并返回片段数。"""
        self.delete_document(document.id)
        self._documents[document.id] = document
        chunks = split_document(document)
        vectors = embed_chunk_batch(self.embedding_provider, [chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            if self.load_chunks_into_memory:
                self._chunks[chunk.id] = chunk
                self._vectors[chunk.embedding_id] = vector
            if chunk.chunk_type == "child":
                self.vector_index.upsert(chunk, vector)
        if self.keyword_index:
            self.keyword_index.upsert_chunks(chunks)
        self.cache_store.clear()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """删除知识文档及相关片段。"""
        self._documents.pop(doc_id, None)
        for chunk_id, chunk in list(self._chunks.items()):
            if chunk.doc_id == doc_id:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk.embedding_id, None)
        self.vector_index.delete_by_doc(doc_id)
        if self.keyword_index:
            self.keyword_index.delete_by_doc(doc_id)
        self.cache_store.clear()

    def rebuild(self, documents: list[KnowledgeDocument]) -> int:
        """使用文档集合重建知识索引。"""
        self._documents.clear()
        self._chunks.clear()
        self._vectors.clear()
        self.vector_index.clear()
        if self.keyword_index:
            self.keyword_index.clear()
        count = 0
        for document in documents:
            count += self.index_document(document)
        self.cache_store.clear()
        return count

    def load_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """Load persisted document/chunk metadata without re-embedding.

        Startup should not call external embedding services or rewrite Milvus.
        Indexing endpoints and workers are responsible for creating vectors.
        """
        self._documents = {document.id: document for document in documents}
        self._chunks = {chunk.id: chunk for chunk in chunks} if self.load_chunks_into_memory else {}
        self._vectors.clear()
        self.cache_store.clear()
        return len(chunks)

    def document_chunks(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """返回文档切分后的片段列表。"""
        return split_document(document)

    def retrieve(self, query: str, filters: dict[str, Any], *, top_k: int = 4) -> RetrievalResult:
        """按查询和过滤条件执行混合检索。"""
        normalized_query = normalize_text(query)
        effective_filter = normalize_retrieval_filter(filters)
        cache_key = self._cache_key(normalized_query, effective_filter, top_k)
        cached_payload = self.cache_store.get_json(cache_key)
        if cached_payload:
            return self._result_from_cache(cached_payload, effective_filter)
        if self._remote_retrieval_enabled():
            result = self._retrieve_remote(normalized_query, effective_filter, top_k=top_k)
            self._cache_result(cache_key, result)
            return result

        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, effective_filter)]
        prefilter_child_count = sum(1 for chunk in self._chunks.values() if chunk.chunk_type == "child")
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        if not candidates:
            result = RetrievalResult(
                sources=[],
                filter=effective_filter,
                status={
                    "query": normalized_query,
                    "prefilter_child_count": prefilter_child_count,
                    "candidate_count": 0,
                    "metadata_coverage": coverage,
                    "cache": {"hit": False, "backend": self.cache_store.backend},
                    "embedding_provider": self.embedding_provider.name,
                    "rerank_provider": self.reranker.name,
                    "metadata_filtering": bool(effective_filter),
                    "channels": {"dense": 0, "bm25": 0, "rrf": 0, "rerank": 0},
                    "filter": effective_filter,
                },
            )
            self._cache_result(cache_key, result)
            return result

        degraded: list[dict[str, str]] = []
        try:
            dense_hits = self._dense_retrieve(normalized_query, candidates, effective_filter, limit=max(top_k * 3, 8))
        except Exception as exc:
            dense_hits = []
            degraded.append({"stage": "dense_retrieve", "error": type(exc).__name__})
        bm25_hits = bm25_retrieve(normalized_query, candidates, limit=max(top_k * 3, 8))
        fused_hits = rrf_fuse([dense_hits, bm25_hits], rrf_k=60)
        try:
            reranked = self.reranker.rerank(normalized_query, fused_hits, effective_filter, top_n=max(top_k, 1))
        except Exception as exc:
            reranked = HeuristicReranker().rerank(normalized_query, fused_hits, effective_filter, top_n=max(top_k, 1))
            degraded.append({"stage": "rerank", "error": type(exc).__name__})
        expanded_sources = expand_parent_context(reranked, self._chunks)
        expanded_sources, structured_backfill_status = expand_structured_section_context(
            normalized_query,
            expanded_sources,
            self._chunks,
        )
        result = RetrievalResult(
            sources=expanded_sources,
            filter=effective_filter,
            status={
                "query": normalized_query,
                "prefilter_child_count": prefilter_child_count,
                "candidate_count": len(candidates),
                "metadata_coverage": coverage,
                "cache": {"hit": False, "backend": self.cache_store.backend},
                "vector_backend": self.vector_index.backend,
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "metadata_filtering": bool(effective_filter),
                "channels": {
                    "dense": len(dense_hits),
                    "bm25": len(bm25_hits),
                    "rrf": len(fused_hits),
                    "rerank": len(reranked),
                },
                "structured_backfill": structured_backfill_status,
                "degraded": degraded,
                "filter": effective_filter,
            },
        )
        self._cache_result(cache_key, result)
        return result

    def retrieve_many(self, queries: list[dict[str, Any]], filters: dict[str, Any]) -> list[RetrievalResult]:
        """批量执行多路检索，合并 embedding 和向量库请求。"""
        effective_filter = normalize_retrieval_filter(filters)
        query_specs = []
        results: list[RetrievalResult | None] = [None] * len(queries)
        for index, query in enumerate(queries):
            text = normalize_text(str(query.get("text") or ""))
            if not text:
                results[index] = self._empty_result("", effective_filter, top_k=int(query.get("top_k") or 3))
                continue
            top_k = int(query.get("top_k") or 3)
            cache_key = self._cache_key(text, effective_filter, top_k)
            cached_payload = self.cache_store.get_json(cache_key)
            if cached_payload:
                results[index] = self._result_from_cache(cached_payload, effective_filter)
            else:
                query_specs.append({"index": index, "query": text, "top_k": top_k, "cache_key": cache_key})

        if not query_specs:
            return [result for result in results if result is not None]
        if self._remote_retrieval_enabled():
            return self._retrieve_many_remote(query_specs, results, effective_filter)

        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, effective_filter)]
        prefilter_child_count = sum(1 for chunk in self._chunks.values() if chunk.chunk_type == "child")
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

        limits = [max(int(spec["top_k"]) * 3, 8) for spec in query_specs]
        dense_by_index: dict[int, list[RetrievedSource]] = {}
        degraded_by_index: dict[int, list[dict[str, str]]] = {int(spec["index"]): [] for spec in query_specs}
        try:
            query_vectors = self.embedding_provider.embed_documents([str(spec["query"]) for spec in query_specs])
            batch_hits = self.vector_index.search_batch(query_vectors, effective_filter, limit=max(limits))
            by_id = {chunk.id: chunk for chunk in candidates}
            for spec, vector_hits, limit in zip(query_specs, batch_hits, limits):
                dense_by_index[int(spec["index"])] = [
                    RetrievedSource(
                        chunk=by_id[chunk_id],
                        score=score,
                        dense_score=score,
                        retrieval_channel="dense_batch",
                    )
                    for chunk_id, score in vector_hits[:limit]
                    if chunk_id in by_id and score > 0
                ]
        except Exception as exc:
            for spec in query_specs:
                degraded_by_index[int(spec["index"])].append({"stage": "dense_retrieve_batch", "error": type(exc).__name__})

        for spec, limit in zip(query_specs, limits):
            index = int(spec["index"])
            query = str(spec["query"])
            top_k = int(spec["top_k"])
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
                top_k=top_k,
                degraded=degraded_by_index.get(index) or [],
                batch_mode=True,
            )
            self._cache_result(str(spec["cache_key"]), result)
            results[index] = result
        return [result for result in results if result is not None]

    def _remote_retrieval_enabled(self) -> bool:
        """Return whether retrieval should hydrate chunks from an external store."""
        return self.chunk_store is not None and self.keyword_index is not None and not self.load_chunks_into_memory

    def _retrieve_remote(self, normalized_query: str, effective_filter: dict[str, Any], *, top_k: int) -> RetrievalResult:
        """Retrieve through external vector/keyword indexes and hydrate chunks on demand."""
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        limit = max(top_k * 3, 8)
        degraded: list[dict[str, str]] = []
        try:
            dense_hits = self._remote_dense_retrieve(normalized_query, effective_filter, limit=limit, channel="dense")
        except Exception as exc:
            dense_hits = []
            degraded.append({"stage": "dense_retrieve", "error": type(exc).__name__})
        try:
            bm25_hits = self._keyword_retrieve(normalized_query, effective_filter, limit=limit)
        except Exception as exc:
            bm25_hits = []
            degraded.append({"stage": "bm25_retrieve", "error": type(exc).__name__})
        hit_chunks = self._unique_hit_chunks(dense_hits, bm25_hits)
        return self._build_result_from_hits(
            normalized_query,
            effective_filter,
            hit_chunks,
            len(hit_chunks),
            coverage,
            dense_hits,
            bm25_hits,
            top_k=top_k,
            degraded=degraded,
            candidate_source="external_indexes",
        )

    def _retrieve_many_remote(
        self,
        query_specs: list[dict[str, Any]],
        results: list[RetrievalResult | None],
        effective_filter: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Batch remote dense retrieval, then hydrate and rerank each query independently."""
        limits = [max(int(spec["top_k"]) * 3, 8) for spec in query_specs]
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        dense_by_index: dict[int, list[RetrievedSource]] = {}
        degraded_by_index: dict[int, list[dict[str, str]]] = {int(spec["index"]): [] for spec in query_specs}
        try:
            query_vectors = self.embedding_provider.embed_documents([str(spec["query"]) for spec in query_specs])
            batch_hits = self.vector_index.search_batch(query_vectors, effective_filter, limit=max(limits))
            for spec, vector_hits, limit in zip(query_specs, batch_hits, limits):
                dense_by_index[int(spec["index"])] = self._sources_from_scored_ids(
                    vector_hits[:limit],
                    score_kind="dense",
                    channel="dense_batch",
                )
        except Exception as exc:
            for spec in query_specs:
                degraded_by_index[int(spec["index"])].append({"stage": "dense_retrieve_batch", "error": type(exc).__name__})

        for spec, limit in zip(query_specs, limits):
            index = int(spec["index"])
            query = str(spec["query"])
            top_k = int(spec["top_k"])
            dense_hits = dense_by_index.get(index) or []
            try:
                bm25_hits = self._keyword_retrieve(query, effective_filter, limit=limit)
            except Exception as exc:
                bm25_hits = []
                degraded_by_index[index].append({"stage": "bm25_retrieve", "error": type(exc).__name__})
            hit_chunks = self._unique_hit_chunks(dense_hits, bm25_hits)
            result = self._build_result_from_hits(
                query,
                effective_filter,
                hit_chunks,
                len(hit_chunks),
                coverage,
                dense_hits,
                bm25_hits,
                top_k=top_k,
                degraded=degraded_by_index.get(index) or [],
                batch_mode=True,
                candidate_source="external_indexes",
            )
            self._cache_result(str(spec["cache_key"]), result)
            results[index] = result
        return [result for result in results if result is not None]

    def _build_result_from_hits(
        self,
        normalized_query: str,
        effective_filter: dict[str, Any],
        candidates: list[KnowledgeChunk],
        prefilter_child_count: int,
        coverage: dict[str, Any],
        dense_hits: list[RetrievedSource],
        bm25_hits: list[RetrievedSource],
        *,
        top_k: int,
        degraded: list[dict[str, str]] | None = None,
        batch_mode: bool = False,
        candidate_source: str = "memory",
    ) -> RetrievalResult:
        """基于 dense/BM25 命中构造标准检索结果。"""
        degraded = list(degraded or [])
        fused_hits = rrf_fuse([dense_hits, bm25_hits], rrf_k=60)
        try:
            reranked = self.reranker.rerank(normalized_query, fused_hits, effective_filter, top_n=max(top_k, 1))
        except Exception as exc:
            reranked = HeuristicReranker().rerank(normalized_query, fused_hits, effective_filter, top_n=max(top_k, 1))
            degraded.append({"stage": "rerank", "error": type(exc).__name__})
        expanded_sources = self._expand_parent_context(reranked)
        expanded_sources, structured_backfill_status = self._expand_structured_section_context(
            normalized_query,
            expanded_sources,
        )
        return RetrievalResult(
            sources=expanded_sources,
            filter=effective_filter,
            status={
                "query": normalized_query,
                "prefilter_child_count": prefilter_child_count,
                "candidate_count": len(candidates),
                "metadata_coverage": coverage,
                "cache": {"hit": False, "backend": self.cache_store.backend},
                "vector_backend": self.vector_index.backend,
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "metadata_filtering": bool(effective_filter),
                "channels": {
                    "dense": len(dense_hits),
                    "bm25": len(bm25_hits),
                    "rrf": len(fused_hits),
                    "rerank": len(reranked),
                },
                "structured_backfill": structured_backfill_status,
                "degraded": degraded,
                "filter": effective_filter,
                "batch_mode": batch_mode,
                "candidate_source": candidate_source,
            },
        )

    def _remote_dense_retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        *,
        limit: int,
        channel: str,
    ) -> list[RetrievedSource]:
        """Run vector retrieval against the configured vector index and hydrate chunks."""
        query_vector = self.embedding_provider.embed_query(query)
        return self._sources_from_scored_ids(
            self.vector_index.search(query_vector, filters, limit=limit),
            score_kind="dense",
            channel=channel,
        )

    def _keyword_retrieve(self, query: str, filters: dict[str, Any], *, limit: int) -> list[RetrievedSource]:
        """Run keyword retrieval against Elasticsearch or the in-memory fallback."""
        if self.keyword_index:
            return self._sources_from_scored_ids(
                self.keyword_index.search(query, filters, limit=limit),
                score_kind="bm25",
                channel=f"{self.keyword_index.backend}_bm25",
            )
        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, filters)]
        return bm25_retrieve(query, candidates, limit=limit)

    def _sources_from_scored_ids(
        self,
        scored_ids: list[tuple[str, float]],
        *,
        score_kind: str,
        channel: str,
    ) -> list[RetrievedSource]:
        """Hydrate scored chunk ids into RetrievedSource objects."""
        if not scored_ids:
            return []
        chunks = self._load_chunks_by_ids([chunk_id for chunk_id, _ in scored_ids])
        by_id = {chunk.id: chunk for chunk in chunks}
        sources = []
        for chunk_id, score in scored_ids:
            chunk = by_id.get(str(chunk_id))
            if chunk is None or score <= 0:
                continue
            sources.append(
                RetrievedSource(
                    chunk=chunk,
                    score=float(score),
                    dense_score=float(score) if score_kind == "dense" else 0.0,
                    bm25_score=float(score) if score_kind == "bm25" else 0.0,
                    retrieval_channel=channel,
                )
            )
        return sources

    def _load_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """Load chunks from memory or the configured persistent chunk store."""
        unique_ids = unique_ordered_strings(chunk_ids)
        if not unique_ids:
            return []
        if self._remote_retrieval_enabled() and self.chunk_store is not None:
            chunks = self.chunk_store.get_chunks_by_ids(unique_ids)
        else:
            chunks = [self._chunks[chunk_id] for chunk_id in unique_ids if chunk_id in self._chunks]
        by_id = {chunk.id: chunk for chunk in chunks}
        return [by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id]

    def _unique_hit_chunks(self, *hit_lists: list[RetrievedSource]) -> list[KnowledgeChunk]:
        """Return unique chunks from retrieval hits in hit order."""
        seen = set()
        chunks = []
        for hits in hit_lists:
            for hit in hits:
                if hit.chunk.id in seen:
                    continue
                seen.add(hit.chunk.id)
                chunks.append(hit.chunk)
        return chunks

    def _expand_parent_context(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        """Expand child hits with parent chunks from memory or persistent storage."""
        if not self._remote_retrieval_enabled() or self.chunk_store is None:
            return expand_parent_context(sources, self._chunks)
        parent_ids = [
            source.chunk.parent_id
            for source in sources
            if source.chunk.chunk_type == "child" and source.chunk.parent_id
        ]
        parents = {chunk.id: chunk for chunk in self.chunk_store.get_parent_chunks(unique_ordered_strings(parent_ids))}
        return expand_parent_context(sources, parents)

    def _expand_structured_section_context(
        self,
        query: str,
        sources: list[RetrievedSource],
    ) -> tuple[list[RetrievedSource], dict[str, Any]]:
        """Backfill structured section siblings from memory or persistent storage."""
        if not self._remote_retrieval_enabled() or self.chunk_store is None:
            return expand_structured_section_context(query, sources, self._chunks)
        target_chunk_type = structured_backfill_chunk_type(query)
        if not target_chunk_type:
            return sources, {"applied": False}
        parent_section_ids = [
            str(source.chunk.metadata.get("parent_section_id") or "")
            for source in sources
            if structured_source_matches(source, target_chunk_type)
        ]
        sibling_chunks = self.chunk_store.get_structured_section_chunks(
            unique_ordered_strings(parent_section_ids),
            target_chunk_type,
        )
        sibling_lookup = {chunk.id: chunk for chunk in sibling_chunks}
        return expand_structured_section_context(query, sources, sibling_lookup)

    def _dense_retrieve(
        self,
        query: str,
        candidates: list[KnowledgeChunk],
        filters: dict[str, Any],
        *,
        limit: int,
    ) -> list[RetrievedSource]:
        """执行实例级向量召回。"""
        query_vector = self.embedding_provider.embed_query(query)
        by_id = {chunk.id: chunk for chunk in candidates}
        vector_hits = self.vector_index.search(query_vector, filters, limit=limit)
        hits = [
            RetrievedSource(
                chunk=by_id[chunk_id],
                score=score,
                dense_score=score,
                retrieval_channel="dense",
            )
            for chunk_id, score in vector_hits
            if chunk_id in by_id and score > 0
        ]
        if hits:
            return hits[:limit]
        if not self._vectors:
            return []
        return dense_retrieve(query, candidates, self._vectors, limit=limit, embedding_provider=self.embedding_provider)

    def _cache_key(self, normalized_query: str, effective_filter: dict[str, Any], top_k: int) -> str:
        """构造单次检索缓存 key。"""
        payload = json.dumps(
            {"query": normalized_query, "filters": effective_filter, "top_k": top_k},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return f"declaration_qa:rag:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _result_from_cache(self, cached_payload: dict[str, Any], effective_filter: dict[str, Any]) -> RetrievalResult:
        """从缓存载荷恢复检索结果。"""
        cached_items = list(cached_payload.get("sources") or [])
        if self._remote_retrieval_enabled():
            chunks = self._load_chunks_by_ids([str(item.get("chunk_id") or "") for item in cached_items])
            by_id = {chunk.id: chunk for chunk in chunks}
        else:
            by_id = self._chunks
        cached_sources = [
            RetrievedSource(
                chunk=by_id[item["chunk_id"]],
                score=float(item.get("score") or 0.0),
                dense_score=float(item.get("dense_score") or 0.0),
                bm25_score=float(item.get("bm25_score") or 0.0),
                rerank_score=float(item.get("rerank_score") or 0.0),
                retrieval_channel=str(item.get("retrieval_channel") or "cache"),
            )
            for item in cached_items
            if item.get("chunk_id") in by_id
        ]
        status = cached_payload.get("status") or {}
        return RetrievalResult(
            sources=cached_sources,
            filter=cached_payload.get("filter") or effective_filter,
            status={**status, "cache": {**(status.get("cache") or {}), "hit": True, "backend": self.cache_store.backend}},
        )

    def _empty_result(
        self,
        normalized_query: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        prefilter_child_count: int | None = None,
        coverage: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """构造空检索结果。"""
        if prefilter_child_count is None:
            prefilter_child_count = sum(1 for chunk in self._chunks.values() if chunk.chunk_type == "child")
        coverage = coverage or metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        return RetrievalResult(
            sources=[],
            filter=effective_filter,
            status={
                "query": normalized_query,
                "top_k": top_k,
                "prefilter_child_count": prefilter_child_count,
                "candidate_count": 0,
                "metadata_coverage": coverage,
                "cache": {"hit": False, "backend": self.cache_store.backend},
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "metadata_filtering": bool(effective_filter),
                "channels": {"dense": 0, "bm25": 0, "rrf": 0, "rerank": 0},
                "filter": effective_filter,
            },
        )

    def _cache_result(self, cache_key: str, result: RetrievalResult) -> None:
        """将检索结果写入缓存。"""
        self.cache_store.set_json(
            cache_key,
            {
                "sources": [source.payload() for source in result.sources],
                "filter": result.filter,
                "status": result.status,
            },
            ttl_seconds=self.cache_ttl_seconds,
        )


def split_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    """按文档配置选择切分策略。"""
    strategy = normalize_chunking_strategy(document.metadata.get("chunking_strategy"))
    if strategy == CHUNKING_HIERARCHICAL:
        return split_hierarchical(document)
    return split_parent_child(document)


def embed_chunk_batch(embedding_provider: EmbeddingProvider, texts: list[str], *, batch_size: int = 10) -> list[list[float]]:
    """批量生成知识片段向量。"""
    vectors: list[list[float]] = []
    for index in range(0, len(texts), batch_size):
        vectors.extend(embedding_provider.embed_documents(texts[index : index + batch_size]))
    return vectors


def normalize_chunking_strategy(value: Any) -> str:
    """标准化文档切分策略名称。"""
    strategy = str(value or CHUNKING_PARENT_CHILD).strip().lower().replace("-", "_")
    return strategy if strategy in SUPPORTED_CHUNKING_STRATEGIES else CHUNKING_PARENT_CHILD


def split_parent_child(document: KnowledgeDocument, *, parent_size: int = 1200, child_size: int = 420, overlap: int = 80) -> list[KnowledgeChunk]:
    """按父子窗口策略切分文档。"""
    content = document.content.strip()
    if not content:
        return []
    parent_texts = split_windows(content, parent_size, max(0, min(overlap, parent_size // 2)))
    chunks: list[KnowledgeChunk] = []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{document.id}:p:{parent_index}"
        parent_chunk = knowledge_chunk_from_document(
            document,
            chunk_id=parent_id,
            parent_id=parent_id,
            chunk_type="parent",
            content=parent_text,
        )
        chunks.append(parent_chunk)
        child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
        for child_index, child_text in enumerate(child_texts):
            child_id = f"{parent_id}:c:{child_index}"
            chunks.append(
                knowledge_chunk_from_document(
                    document,
                    chunk_id=child_id,
                    parent_id=parent_id,
                    chunk_type="child",
                    content=child_text,
                )
            )
    return chunks


def split_hierarchical(document: KnowledgeDocument, *, parent_size: int = 1200, child_size: int = 420, overlap: int = 80) -> list[KnowledgeChunk]:
    """按标题层级切分结构化文档。"""
    content = document.content.strip()
    if not content:
        return []
    chunks: list[KnowledgeChunk] = []
    for section_index, section in enumerate(markdown_sections(content)):
        base_metadata = hierarchical_section_metadata(document, section, section_index)
        section_content = section["content"].strip()
        if len(section_content) <= child_size:
            child_id = f"{document.id}:h:{section_index}:c:0"
            chunks.append(
                knowledge_chunk_from_document(
                    document,
                    chunk_id=child_id,
                    parent_id="",
                    chunk_type="child",
                    content=section_content,
                    extra_metadata={
                        **base_metadata,
                        "child_window_index": 0,
                        "indexable": True,
                    },
                )
            )
            continue

        parent_texts = split_windows(section_content, parent_size, max(0, min(overlap, parent_size // 2)))
        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = f"{document.id}:h:{section_index}:p:{parent_index}"
            child_texts = split_windows(parent_text, child_size, max(0, min(overlap, child_size // 2)))
            if len(child_texts) == 1 and child_texts[0].strip() == parent_text.strip():
                child_id = f"{document.id}:h:{section_index}:p:{parent_index}:c:0"
                chunks.append(
                    knowledge_chunk_from_document(
                        document,
                        chunk_id=child_id,
                        parent_id="",
                        chunk_type="child",
                        content=parent_text,
                        extra_metadata={
                            **base_metadata,
                            "parent_window_index": parent_index,
                            "child_window_index": 0,
                            "indexable": True,
                        },
                    )
                )
                continue

            parent_chunk = knowledge_chunk_from_document(
                document,
                chunk_id=parent_id,
                parent_id=parent_id,
                chunk_type="parent",
                content=parent_text,
                extra_metadata={
                    **base_metadata,
                    "parent_window_index": parent_index,
                    "context_role": "section_context",
                    "indexable": False,
                    "context_only": True,
                },
            )
            chunks.append(parent_chunk)
            for child_index, child_text in enumerate(child_texts):
                child_id = f"{parent_id}:c:{child_index}"
                chunks.append(
                    knowledge_chunk_from_document(
                        document,
                        chunk_id=child_id,
                        parent_id=parent_id,
                        chunk_type="child",
                        content=child_text,
                        extra_metadata={
                            **base_metadata,
                            "parent_window_index": parent_index,
                            "child_window_index": child_index,
                            "indexable": True,
                        },
                    )
                )
    return chunks


def hierarchical_section_metadata(document: KnowledgeDocument, section: dict[str, Any], section_index: int) -> dict[str, Any]:
    """为层级片段生成结构化元数据。"""
    section_title = str(section.get("section_title") or section.get("title") or "Document")
    node_title = str(section.get("node_title") or section.get("title") or section_title)
    node_order = parse_heading_order(node_title)
    if node_order is None:
        node_order = section_index
    semantic_chunk_type = classify_hierarchical_chunk_type(section_title, node_title)
    return {
        "chunking_strategy": CHUNKING_HIERARCHICAL,
        "heading_path": section["path"],
        "heading_level": section["level"],
        "section_index": section_index,
        "section_title": section_title,
        "node_title": node_title,
        "parent_section_id": f"{document.id}#{section_title}",
        "node_order": node_order,
        "chunk_type": semantic_chunk_type,
    }


def classify_hierarchical_chunk_type(section_title: str, node_title: str) -> str:
    """识别层级片段的结构类型。"""
    if section_title == "申请材料" and parse_heading_order(node_title) is not None:
        return "material_item"
    normalized_node_title = strip_heading_order(node_title)
    if section_title == "办理流程" and normalized_node_title in {"受理", "审查", "决定", "办结", "送达"}:
        return "process_step"
    return "section_leaf"


def parse_heading_order(title: str) -> int | None:
    """解析标题中的序号层级。"""
    match = re.match(r"^\s*(\d+)(?:[\.、．\)]\s*|\s+)", str(title or ""))
    if not match:
        return None
    return int(match.group(1))


def strip_heading_order(title: str) -> str:
    """移除标题前缀序号。"""
    return re.sub(r"^\s*\d+(?:[\.、．\)]\s*|\s+)", "", str(title or "")).strip()


def markdown_sections(text: str) -> list[dict[str, Any]]:
    """将 Markdown 文本解析为标题章节。"""
    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    current_level = 0
    current_title = ""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def flush_current() -> None:
        """刷新当前累积的章节内容。"""
        body = "\n".join(current_lines).strip()
        if not body:
            return
        path = " > ".join(current_path) if current_path else "Document"
        section_text = f"{path}\n\n{body}".strip() if current_path else body
        section_title = markdown_section_title(stack, current_title)
        node_title = current_title or section_title
        sections.append(
            {
                "path": path,
                "level": current_level,
                "title": current_title,
                "section_title": section_title,
                "node_title": node_title,
                "body": body,
                "content": section_text,
            }
        )

    for line in text.splitlines():
        heading = heading_pattern.match(line.strip())
        if heading:
            flush_current()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_path = [item[1] for item in stack]
            current_level = level
            current_title = title
            current_lines = []
            continue
        current_lines.append(line)
    flush_current()

    if sections:
        return [section for section in sections if section["content"].strip()]
    return [{"path": "Document", "level": 0, "content": text.strip()}]


def markdown_section_title(stack: list[tuple[int, str]], current_title: str) -> str:
    """读取 Markdown 标题文本。"""
    for level, title in stack:
        if level == 2:
            return title
    if current_title:
        return current_title
    if stack:
        return stack[-1][1]
    return "Document"


def split_windows(text: str, size: int, overlap: int) -> list[str]:
    """将文本按窗口和重叠大小切分。"""
    content = str(text or "").strip()
    if len(content) <= size:
        return [content]
    units = split_window_units(content, max(1, size))
    output = []
    current: list[str] = []
    for unit in units:
        candidate = join_window_units([*current, unit])
        if current and len(candidate) > size:
            output.append(join_window_units(current))
            current = trailing_overlap_units(current, overlap)
        current.append(unit)
    if current:
        output.append(join_window_units(current))
    return [item for item in output if item]


def split_window_units(text: str, size: int) -> list[str]:
    """将文本拆分为适合滑窗的基础单元。"""
    units: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        units.extend(split_long_line(line.rstrip(), size))
    return units


def split_long_line(line: str, size: int) -> list[str]:
    """按长度拆分过长文本行。"""
    if len(line) <= size:
        return [line]
    if "http://" in line or "https://" in line:
        return [line]
    parts: list[str] = []
    remaining = line.strip()
    while len(remaining) > size:
        split_at = best_soft_split(remaining, size)
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def best_soft_split(text: str, size: int) -> int:
    """在目标长度附近寻找软切分点。"""
    search_start = max(1, size // 2)
    candidates = [text.rfind(separator, search_start, size) for separator in ["。", "；", "，", "、", " ", "\t"]]
    split_at = max(candidates)
    if split_at <= 0:
        return size
    return split_at + 1


def trailing_overlap_units(units: list[str], overlap: int) -> list[str]:
    """计算滑窗重叠所需的尾部单元。"""
    if overlap <= 0:
        return []
    retained: list[str] = []
    total = 0
    for unit in reversed(units):
        total += len(unit)
        if retained and total > overlap:
            break
        retained.insert(0, unit)
    return retained


def join_window_units(units: list[str]) -> str:
    """合并滑窗单元为文本片段。"""
    return "\n".join(unit for unit in units if unit.strip()).strip()


def knowledge_chunk_from_document(
    document: KnowledgeDocument,
    *,
    chunk_id: str,
    parent_id: str,
    chunk_type: str,
    content: str,
    extra_metadata: dict[str, Any] | None = None,
) -> KnowledgeChunk:
    """由文档内容创建知识片段对象。"""
    document_metadata = sanitize_metadata(document.metadata)
    metadata = {
        **document_metadata,
        "region_code": document.region_code,
        "region_key": document_metadata.get("region_key") or document.region_code,
        "policy_version": document.policy_version,
        "source_version": document_metadata.get("source_version") or document.policy_version,
        "audit_node": document.audit_node,
        "version_group_uid": document.version_group_uid or document.id,
        "version_no": document.version_no,
        "rollout_percent": document.rollout_percent,
        "doc_status": document_metadata.get("doc_status") or document.status,
    }
    metadata.update(sanitize_metadata(extra_metadata or {}))
    return KnowledgeChunk(
        id=chunk_id,
        doc_id=document.id,
        parent_id=parent_id,
        chunk_type=chunk_type,
        title=document.title,
        content=content,
        region_code=document.region_code,
        policy_version=document.policy_version,
        source_url=document.source_url,
        audit_node=document.audit_node,
        status=document.status,
        effective_start_time=document.effective_start_time,
        effective_end_time=document.effective_end_time,
        embedding_id=f"vec:{chunk_id}",
        metadata=metadata,
    )


def dense_retrieve(
    query: str,
    candidates: list[KnowledgeChunk],
    vectors: dict[str, list[float]],
    *,
    limit: int,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievedSource]:
    """执行向量召回并转换为检索来源。"""
    embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=EMBEDDING_DIMS)
    query_vector = embedding_provider.embed_query(query)
    hits = []
    for chunk in candidates:
        vector = vectors.get(chunk.embedding_id)
        if not vector:
            continue
        score = cosine(query_vector, vector)
        if score > 0:
            hits.append(
                RetrievedSource(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    retrieval_channel="dense",
                )
            )
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def expand_parent_context(sources: list[RetrievedSource], chunks: dict[str, KnowledgeChunk]) -> list[RetrievedSource]:
    """为子片段补充父级上下文。"""
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
    return expanded


def expand_structured_section_context(
    query: str,
    sources: list[RetrievedSource],
    chunks: dict[str, KnowledgeChunk],
) -> tuple[list[RetrievedSource], dict[str, Any]]:
    """为结构化片段补充章节上下文。"""
    target_chunk_type = structured_backfill_chunk_type(query)
    if not target_chunk_type:
        return sources, {"applied": False}

    expanded: list[RetrievedSource] = []
    seen_chunk_ids: set[str] = set()
    seen_node_keys: set[tuple[str, str, str]] = set()
    backfilled_sections: set[str] = set()
    added_count = 0

    for source in sources:
        parent_section_id = str(source.chunk.metadata.get("parent_section_id") or "")
        if (
            parent_section_id
            and parent_section_id not in backfilled_sections
            and structured_source_matches(source, target_chunk_type)
        ):
            siblings = structured_section_sibling_sources(source, chunks, parent_section_id, target_chunk_type)
            for sibling in siblings:
                if append_unique_source(expanded, sibling, seen_chunk_ids, seen_node_keys):
                    added_count += 1
            backfilled_sections.add(parent_section_id)
        append_unique_source(expanded, source, seen_chunk_ids, seen_node_keys)

    return expanded, {
        "applied": bool(backfilled_sections),
        "target_chunk_type": target_chunk_type,
        "section_count": len(backfilled_sections),
        "added_source_count": added_count,
    }


def structured_backfill_chunk_type(query: str) -> str:
    """为结构化来源补齐片段类型。"""
    text = str(query or "")
    if any(keyword in text for keyword in ["材料", "申请材料", "要交啥", "要准备什么", "提交什么", "上传什么", "需要哪些材料", "要带", "需要什么"]):
        return "material_item"
    if any(keyword in text for keyword in ["流程", "怎么办", "怎么申请", "办理步骤", "提交后", "审核", "如何办理"]):
        return "process_step"
    return ""


def structured_source_matches(source: RetrievedSource, target_chunk_type: str) -> bool:
    """判断结构化来源是否与目标片段匹配。"""
    metadata = source.chunk.metadata or {}
    semantic_chunk_type = str(metadata.get("chunk_type") or "")
    section_title = str(metadata.get("section_title") or "")
    if semantic_chunk_type == target_chunk_type:
        return True
    if target_chunk_type == "material_item":
        return section_title == "申请材料"
    if target_chunk_type == "process_step":
        return section_title == "办理流程"
    return False


def structured_section_sibling_sources(
    trigger: RetrievedSource,
    chunks: dict[str, KnowledgeChunk],
    parent_section_id: str,
    target_chunk_type: str,
) -> list[RetrievedSource]:
    """查找同一结构章节下的兄弟来源。"""
    grouped: dict[tuple[int, str], list[KnowledgeChunk]] = defaultdict(list)
    for chunk in chunks.values():
        metadata = chunk.metadata or {}
        if chunk.chunk_type != "child":
            continue
        if metadata.get("indexable") is False:
            continue
        if str(metadata.get("parent_section_id") or "") != parent_section_id:
            continue
        if str(metadata.get("chunk_type") or "") != target_chunk_type:
            continue
        node_title = str(metadata.get("node_title") or "")
        node_order = safe_int(metadata.get("node_order"))
        grouped[(node_order, node_title)].append(chunk)

    sibling_sources = []
    for _, node_chunks in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        sibling_sources.append(merged_structured_node_source(trigger, node_chunks))
    return sibling_sources


def merged_structured_node_source(trigger: RetrievedSource, node_chunks: list[KnowledgeChunk]) -> RetrievedSource:
    """合并结构化节点上下文为单个来源。"""
    ordered_chunks = sorted(node_chunks, key=structured_chunk_sort_key)
    first = ordered_chunks[0]
    merged_content = merge_structured_node_content(ordered_chunks)
    merged_chunk = replace(
        first,
        id=first.id,
        parent_id="",
        content=merged_content,
        metadata={
            **first.metadata,
            "merged_chunk_ids": [chunk.id for chunk in ordered_chunks],
            "merged_chunk_count": len(ordered_chunks),
            "structured_backfill_merged": True,
        },
    )
    return RetrievedSource(
        chunk=merged_chunk,
        score=trigger.score,
        dense_score=0.0,
        bm25_score=0.0,
        rerank_score=0.0,
        retrieval_channel=f"{trigger.retrieval_channel}+structured_backfill",
    )


def structured_chunk_sort_key(chunk: KnowledgeChunk) -> tuple[int, int, int, str]:
    """生成结构化片段排序键。"""
    metadata = chunk.metadata or {}
    return (
        safe_int(metadata.get("node_order")),
        safe_int(metadata.get("parent_window_index")),
        safe_int(metadata.get("child_window_index")),
        chunk.id,
    )


def merge_structured_node_content(chunks: list[KnowledgeChunk]) -> str:
    """合并结构化节点及相关内容。"""
    parts: list[str] = []
    seen_content: set[str] = set()
    for chunk in chunks:
        content = chunk.content.strip()
        if not content or content in seen_content:
            continue
        parts.append(content)
        seen_content.add(content)
    return "\n\n".join(parts)


def append_unique_source(
    output: list[RetrievedSource],
    source: RetrievedSource,
    seen_chunk_ids: set[str],
    seen_node_keys: set[tuple[str, str, str]],
) -> bool:
    """按片段 ID 追加唯一来源。"""
    chunk = source.chunk
    if chunk.id in seen_chunk_ids:
        return False
    metadata = chunk.metadata or {}
    parent_section_id = str(metadata.get("parent_section_id") or "")
    semantic_chunk_type = str(metadata.get("chunk_type") or "")
    node_title = str(metadata.get("node_title") or "")
    if parent_section_id and semantic_chunk_type and node_title:
        node_key = (parent_section_id, semantic_chunk_type, node_title)
        if node_key in seen_node_keys:
            return False
        seen_node_keys.add(node_key)
    seen_chunk_ids.add(chunk.id)
    output.append(source)
    return True


def safe_int(value: Any) -> int:
    """安全地将值转换为整数。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def bm25_retrieve(query: str, candidates: list[KnowledgeChunk], *, limit: int) -> list[RetrievedSource]:
    """执行 BM25 关键词召回。"""
    query_tokens = tokenize(query)
    corpus = [tokenize(chunk.content) for chunk in candidates]
    if not query_tokens:
        return []
    scores = bm25_scores(query_tokens, corpus)
    max_score = max(scores or [0.0]) or 1.0
    hits = []
    for chunk, score in zip(candidates, scores, strict=True):
        normalized_score = float(score) / max_score if score > 0 else 0.0
        if normalized_score <= 0:
            continue
        hits.append(
            RetrievedSource(
                chunk=chunk,
                score=normalized_score,
                bm25_score=normalized_score,
                retrieval_channel="bm25",
            )
        )
    return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]


def bm25_scores(query_tokens: list[str], corpus: list[list[str]]) -> list[float]:
    """计算候选片段的 BM25 分数。"""
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
    """使用 RRF 融合多路召回结果。"""
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


def rerank(query: str, hits: list[RetrievedSource], filters: dict[str, Any], *, top_n: int) -> list[RetrievedSource]:
    """按查询相关性重新排序候选来源。"""
    query_tokens = set(tokenize(query))
    reranked = []
    for hit in hits:
        content_tokens = set(tokenize(hit.chunk.content))
        overlap = len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))
        metadata_boost = metadata_match_boost(hit.chunk, filters)
        rerank_score = min(1.0, 0.55 * overlap + 0.35 * hit.dense_score + 0.25 * hit.bm25_score + metadata_boost)
        reranked.append(
            RetrievedSource(
                chunk=hit.chunk,
                score=max(hit.score, rerank_score),
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
                rerank_score=rerank_score,
                retrieval_channel=f"{hit.retrieval_channel}+rerank",
            )
        )
    return sorted(reranked, key=lambda item: (item.rerank_score, item.score), reverse=True)[:top_n]


def metadata_match_boost(chunk: KnowledgeChunk, filters: dict[str, Any]) -> float:
    """计算片段元数据与过滤条件的加权提升。"""
    if not filters:
        return 0.0
    metadata = chunk.metadata or {}
    matched = sum(
        1
        for key, value in filters.items()
        if key not in {"as_of", "rollout_key"}
        and metadata_value_matches(chunk.status if key == "status" else metadata.get(key), value)
    )
    return min(0.2, matched * 0.05)


def matches_filter(chunk: KnowledgeChunk, filters: dict[str, Any]) -> bool:
    """判断知识片段是否满足检索过滤条件。"""
    if not filters:
        return True
    metadata = chunk.metadata or {}
    for key, value in filters.items():
        if key in {"as_of", "rollout_key"}:
            continue
        candidate = chunk.status if key == "status" else metadata.get(key)
        if not metadata_value_matches(candidate, value):
            return False
    return True


def metadata_value_matches(candidate: Any, expected: Any) -> bool:
    """按字符串或列表语义匹配元数据值。"""
    if isinstance(expected, list):
        return any(metadata_value_matches(candidate, item) for item in expected)
    if isinstance(candidate, list):
        return any(metadata_value_matches(item, expected) for item in candidate)
    return normalize_filter_text(candidate) == normalize_filter_text(expected)


def normalize_filter_text(value: Any) -> str:
    """标准化元数据过滤比较文本。"""
    return normalize_text(str(value or ""))


def unique_ordered_strings(values: list[str]) -> list[str]:
    """Return non-empty strings in first-seen order."""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def parse_filter_date(value: Any) -> date | None:
    """解析检索过滤中的日期值。"""
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def embed_text(text: str) -> list[float]:
    """生成文本的本地哈希向量。"""
    values = [0.0 for _ in range(EMBEDDING_DIMS)]
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % EMBEDDING_DIMS
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    """计算两个向量的余弦点积分数。"""
    left_values = list(left)
    right_values = list(right)
    return sum(a * b for a, b in zip(left_values, right_values))


def normalize_text(value: str) -> str:
    """将文本规范化为小写紧凑形式。"""
    return re.sub(r"\s+", " ", str(value or "").strip())


def tokenize(text: str) -> list[str]:
    """将文本切分为检索用词元。"""
    normalized = normalize_text(text).lower()
    try:
        import jieba

        tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
    except Exception:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalized)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if len(token) >= 4 and re.search(r"[\u4e00-\u9fff]", token):
            expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
    return [token for token in expanded if token]


def build_seed_rag_engine(
    *,
    cache_store: CacheStore | None = None,
    vector_index: VectorIndex | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    keyword_index: KeywordIndex | None = None,
    chunk_store: ChunkStore | None = None,
    load_chunks_into_memory: bool = True,
    cache_ttl_seconds: int = 600,
    include_seed: bool = True,
) -> InMemoryRAGEngine:
    """创建并可选加载种子知识的 RAG 引擎。"""
    engine = InMemoryRAGEngine(
        cache_store=cache_store,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
        reranker=reranker,
        keyword_index=keyword_index,
        chunk_store=chunk_store,
        load_chunks_into_memory=load_chunks_into_memory,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    if include_seed:
        for document in seed_documents():
            engine.index_document(document)
    return engine


def seed_documents() -> list[KnowledgeDocument]:
    """返回内置种子知识文档列表。"""
    return [
        KnowledgeDocument(
            id="doc-residence-guide-2025",
            title="安徽省居住证办理材料指南",
            content=(
                "办理居住证通常需要本人有效身份证件、居住地址证明、近期照片等材料。"
                "居住地址证明可以包括房屋租赁合同、用人单位住宿证明、学校住宿证明或社区出具的居住证明。"
                "申请人应确保材料真实、完整，材料不齐全时需要补正后重新提交。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://guides/residence-permit",
        ),
        KnowledgeDocument(
            id="doc-health-process-2025",
            title="健康证办理流程节点说明",
            content=(
                "健康证申请提交后会进入材料受理、部门审核、制证和办结等节点。"
                "部门审核中表示申请材料已进入业务部门审核环节，用户通常无需重复提交。"
                "如审核发现材料问题，系统会退回并提示补正原因。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://process/health-certificate",
        ),
        KnowledgeDocument(
            id="doc-id-card-photo-rule-2025",
            title="身份证照片上传规范",
            content=(
                "身份证正面照片应保持边框完整、文字清晰、无遮挡、无反光。"
                "如果照片模糊、证件号码无法识别、边缘缺失或存在明显反光，审核人员可退回申请。"
                "用户需要重新上传清晰的身份证正面照片后再次提交。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://rules/id-card-photo",
            audit_node="人工初审",
        ),
        KnowledgeDocument(
            id="doc-residence-proof-rule-2025",
            title="居住证明补正规则",
            content=(
                "居住证明材料缺失时，申请人应补充租赁合同、单位住宿证明、学校住宿证明或社区居住证明。"
                "补交材料需要与当前申请地区一致，并能证明申请人在当地实际居住。"
            ),
            region_code="340000",
            policy_version="2025",
            source_url="internal://rules/residence-proof",
            audit_node="材料初审",
        ),
    ]
