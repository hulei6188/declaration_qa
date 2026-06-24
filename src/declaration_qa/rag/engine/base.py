"""Common RAG engine contracts and orchestration."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.rag.engine.chunking import split_document
from declaration_qa.rag.engine.constants import EMBEDDING_DIMS
from declaration_qa.rag.engine.scoring import rrf_fuse
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from declaration_qa.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider
from declaration_qa.rag.model_providers.rerankers.heuristic_reranker import HeuristicReranker
from declaration_qa.rag.model_providers.rerankers.reranker import Reranker
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage, normalize_retrieval_filter
from declaration_qa.search.keyword import KeywordIndex
from declaration_qa.search.vector.base import VectorIndex
from declaration_qa.storage.cache import CacheStore, InMemoryCacheStore


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


class RAGEngine(Protocol):
    """Common retrieval/indexing contract used by the workflow."""

    cache_store: CacheStore
    vector_index: VectorIndex
    keyword_index: KeywordIndex | None
    chunk_store: ChunkStore | None
    embedding_provider: EmbeddingProvider
    reranker: Reranker

    @property
    def chunk_count(self) -> int:
        """Return locally loaded chunk count."""
        ...

    def documents(self) -> list[KnowledgeDocument]:
        """Return documents known to the engine."""
        ...

    def document_chunks(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Split a document into indexable chunks."""
        ...

    def index_document(self, document: KnowledgeDocument) -> int:
        """Index one document."""
        ...

    def index_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """Index precomputed chunks."""
        ...

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from indexes."""
        ...

    def rebuild(self, documents: list[KnowledgeDocument]) -> int:
        """Rebuild indexes from documents."""
        ...

    def rebuild_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """Rebuild indexes from persisted chunks."""
        ...

    def load_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """Load persisted metadata without rewriting external indexes."""
        ...

    def retrieve(self, query: str, filters: dict[str, Any], *, top_k: int = 4) -> RetrievalResult:
        """Retrieve sources for one query."""
        ...

    def retrieve_many(self, queries: list[dict[str, Any]], filters: dict[str, Any]) -> list[RetrievalResult]:
        """Retrieve sources for multiple rewritten queries."""
        ...

    def retrieve_rewrites(
        self,
        original_query: str,
        retrieval_queries: list[dict[str, Any]],
        filters: dict[str, Any],
        *,
        top_k: int = 6,
    ) -> RetrievalResult:
        """Retrieve for multiple rewrite queries belonging to one user question."""
        ...


class BaseRAGEngine:
    """Shared RAG orchestration for concrete retrieval backends."""

    def __init__(
        self,
        *,
        cache_store: CacheStore | None = None,
        vector_index: VectorIndex,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        keyword_index: KeywordIndex | None = None,
        cache_ttl_seconds: int = 600,
    ) -> None:
        """Initialize shared providers and stores."""
        self.cache_store = cache_store or InMemoryCacheStore()
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=EMBEDDING_DIMS)
        self.reranker = reranker or HeuristicReranker()
        self.keyword_index = keyword_index
        self.chunk_store: ChunkStore | None = None
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))

    def index_document(self, document: KnowledgeDocument) -> int:
        """Index one document."""
        return self.index_chunks(document, split_document(document))

    def rebuild(self, documents: list[KnowledgeDocument]) -> int:
        """Rebuild indexes from documents."""
        return self.rebuild_indexed_data(
            documents,
            [chunk for document in documents for chunk in split_document(document)],
        )

    def document_chunks(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """Return document chunks."""
        return split_document(document)

    def retrieve(self, query: str, filters: dict[str, Any], *, top_k: int = 4) -> RetrievalResult:
        """Retrieve sources for one query."""
        normalized_query = normalize_text(query)
        effective_filter = normalize_retrieval_filter(filters)
        cache_key = self._cache_key(normalized_query, effective_filter, top_k)
        cached_payload = self.cache_store.get_json(cache_key)
        if cached_payload:
            return self._result_from_cache(cached_payload, effective_filter)
        result = self._retrieve_uncached(normalized_query, effective_filter, top_k=top_k)
        self._cache_result(cache_key, result)
        return result

    def retrieve_many(self, queries: list[dict[str, Any]], filters: dict[str, Any]) -> list[RetrievalResult]:
        """Retrieve sources for multiple rewritten queries."""
        effective_filter = normalize_retrieval_filter(filters)
        query_specs = []
        results: list[RetrievalResult | None] = [None] * len(queries)
        for index, query in enumerate(queries):
            text = normalize_text(str(query.get("text") or ""))
            top_k = int(query.get("top_k") or 3)
            if not text:
                results[index] = self._empty_result("", effective_filter, top_k=top_k)
                continue
            cache_key = self._cache_key(text, effective_filter, top_k)
            cached_payload = self.cache_store.get_json(cache_key)
            if cached_payload:
                results[index] = self._result_from_cache(cached_payload, effective_filter)
            else:
                query_specs.append({"index": index, "query": text, "top_k": top_k, "cache_key": cache_key})
        if not query_specs:
            return [result for result in results if result is not None]
        return self._retrieve_many_uncached(query_specs, results, effective_filter)

    def retrieve_rewrites(
        self,
        original_query: str,
        retrieval_queries: list[dict[str, Any]],
        filters: dict[str, Any],
        *,
        top_k: int = 6,
    ) -> RetrievalResult:
        """Retrieve rewrite queries as recall lists, then fuse and rerank once."""
        effective_filter = normalize_retrieval_filter(filters)
        query_specs = []
        for index, query in enumerate(retrieval_queries):
            text = normalize_text(str(query.get("text") or query.get("query") or ""))
            if not text:
                continue
            query_specs.append(
                {
                    "index": index,
                    "query": text,
                    "top_k": int(query.get("top_k") or 3),
                    "id": query.get("id") or "",
                    "strategy": query.get("strategy") or "",
                }
            )
        normalized_query = normalize_text(original_query) or "\n".join(str(spec["query"]) for spec in query_specs)
        top_k = max(1, int(top_k or 6))
        if not query_specs:
            return self._empty_rewrite_result(normalized_query, [], effective_filter, top_k=top_k)
        cache_key = self._rewrite_cache_key(normalized_query, query_specs, effective_filter, top_k)
        cached_payload = self.cache_store.get_json(cache_key)
        if cached_payload:
            return self._result_from_cache(cached_payload, effective_filter)
        result = self._retrieve_rewrites_uncached(normalized_query, query_specs, effective_filter, top_k=top_k)
        self._cache_result(cache_key, result)
        return result

    def _retrieve_uncached(self, normalized_query: str, effective_filter: dict[str, Any], *, top_k: int) -> RetrievalResult:
        """Retrieve without cache."""
        raise NotImplementedError

    def _retrieve_many_uncached(
        self,
        query_specs: list[dict[str, Any]],
        results: list[RetrievalResult | None],
        effective_filter: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Batch retrieve without cache."""
        raise NotImplementedError

    def _retrieve_rewrites_uncached(
        self,
        original_query: str,
        query_specs: list[dict[str, Any]],
        effective_filter: dict[str, Any],
        *,
        top_k: int,
    ) -> RetrievalResult:
        """Retrieve rewrite recall lists without cache."""
        raise NotImplementedError

    def _flush_vector_index(self) -> None:
        """Make newly indexed vector rows visible when supported."""
        flush = getattr(self.vector_index, "flush", None)
        if callable(flush):
            flush()

    def _candidate_source(self) -> str:
        """Return candidate source label."""
        return "memory"

    def _prefilter_child_count(self) -> int:
        """Return backend-specific child count."""
        return 0

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
        candidate_source: str | None = None,
    ) -> RetrievalResult:
        """Build a standard retrieval result from channel hits."""
        degraded = list(degraded or [])
        fused_hits = rrf_fuse([dense_hits, bm25_hits], rrf_k=60)
        try:
            reranked = self.reranker.rerank(normalized_query, fused_hits, effective_filter, top_n=max(top_k, 1))
        except Exception as exc:
            reranked = HeuristicReranker().rerank(normalized_query, fused_hits, effective_filter, top_n=max(top_k, 1))
            degraded.append({"stage": "rerank", "error": type(exc).__name__})
        expanded_sources = self._expand_parent_context(reranked)
        expanded_sources, structured_backfill_status = self._expand_structured_section_context(normalized_query, expanded_sources)
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
                "candidate_source": candidate_source or self._candidate_source(),
            },
        )

    def _build_rewrite_result_from_hit_lists(
        self,
        original_query: str,
        effective_filter: dict[str, Any],
        query_specs: list[dict[str, Any]],
        query_statuses: list[dict[str, Any]],
        hit_lists: list[list[RetrievedSource]],
        coverage: dict[str, Any],
        *,
        top_k: int,
        degraded: list[dict[str, str]] | None = None,
        candidate_count: int | None = None,
        prefilter_child_count: int | None = None,
        candidate_source: str | None = None,
    ) -> RetrievalResult:
        """Build one fused retrieval result for related rewrite queries."""
        degraded = list(degraded or [])
        if not hit_lists:
            return self._empty_rewrite_result(
                original_query,
                query_statuses,
                effective_filter,
                top_k=top_k,
                coverage=coverage,
                degraded=degraded,
                candidate_source=candidate_source,
                prefilter_child_count=prefilter_child_count,
            )
        fused_hits = rrf_fuse(hit_lists, rrf_k=60)
        rerank_candidate_limit = min(len(fused_hits), max(top_k * 10, 40), 100)
        rerank_candidates = fused_hits[:rerank_candidate_limit]
        try:
            reranked = self.reranker.rerank(original_query, rerank_candidates, effective_filter, top_n=max(top_k, 1))
        except Exception as exc:
            reranked = HeuristicReranker().rerank(original_query, rerank_candidates, effective_filter, top_n=max(top_k, 1))
            degraded.append({"stage": "rerank", "error": type(exc).__name__})
        expanded_sources = self._expand_parent_context(reranked)
        expanded_sources, structured_backfill_status = self._expand_structured_section_context(original_query, expanded_sources)
        unique_hit_ids = {
            source.chunk.id
            for hit_list in hit_lists
            for source in hit_list
        }
        return RetrievalResult(
            sources=expanded_sources,
            filter=effective_filter,
            status={
                "query": original_query,
                "query_count": len(query_specs),
                "queries": query_statuses,
                "top_k": top_k,
                "source_count": len(expanded_sources),
                "prefilter_child_count": prefilter_child_count if prefilter_child_count is not None else candidate_count or len(unique_hit_ids),
                "candidate_count": candidate_count if candidate_count is not None else len(unique_hit_ids),
                "metadata_coverage": coverage,
                "cache": {"hit": False, "backend": self.cache_store.backend},
                "vector_backend": self.vector_index.backend,
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "metadata_filtering": bool(effective_filter),
                "channels": {
                    "hit_lists": len(hit_lists),
                    "rrf": len(fused_hits),
                    "rerank_candidates": len(rerank_candidates),
                    "rerank": len(reranked),
                    "dense": sum(int((status.get("channels") or {}).get("dense") or 0) for status in query_statuses),
                    "bm25": sum(int((status.get("channels") or {}).get("bm25") or 0) for status in query_statuses),
                },
                "structured_backfill": structured_backfill_status,
                "degraded": degraded,
                "filter": effective_filter,
                "batch_mode": True,
                "retrieval_mode": "batch_multi_query_rewrite_fused",
                "candidate_source": candidate_source or self._candidate_source(),
            },
        )

    def _sources_from_scored_ids(
        self,
        scored_ids: list[tuple[str, float]],
        *,
        score_kind: str,
        channel: str,
    ) -> list[RetrievedSource]:
        """Hydrate scored chunk ids into sources."""
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

    def _sources_from_scored_ids_lookup(
        self,
        scored_ids: list[tuple[str, float]],
        chunks_by_id: dict[str, KnowledgeChunk],
        *,
        score_kind: str,
        channel: str,
    ) -> list[RetrievedSource]:
        """Hydrate scored ids from a preloaded chunk lookup."""
        sources = []
        for chunk_id, score in scored_ids:
            chunk = chunks_by_id.get(str(chunk_id))
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
        """Load chunks by public ids."""
        raise NotImplementedError

    def _unique_hit_chunks(self, *hit_lists: list[RetrievedSource]) -> list[KnowledgeChunk]:
        """Return unique hit chunks."""
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
        """Expand parent context."""
        raise NotImplementedError

    def _expand_structured_section_context(
        self,
        query: str,
        sources: list[RetrievedSource],
    ) -> tuple[list[RetrievedSource], dict[str, Any]]:
        """Backfill structured section context."""
        raise NotImplementedError

    def _cache_key(self, normalized_query: str, effective_filter: dict[str, Any], top_k: int) -> str:
        """Build a retrieval cache key."""
        payload = json.dumps(
            {"query": normalized_query, "filters": effective_filter, "top_k": top_k},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return f"declaration_qa:rag:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _rewrite_cache_key(
        self,
        normalized_query: str,
        query_specs: list[dict[str, Any]],
        effective_filter: dict[str, Any],
        top_k: int,
    ) -> str:
        """Build a fused rewrite retrieval cache key."""
        payload = json.dumps(
            {
                "query": normalized_query,
                "rewrite_queries": [
                    {
                        "query": spec.get("query") or "",
                        "top_k": int(spec.get("top_k") or 3),
                        "id": spec.get("id") or "",
                        "strategy": spec.get("strategy") or "",
                    }
                    for spec in query_specs
                ],
                "filters": effective_filter,
                "top_k": top_k,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return f"declaration_qa:rag:rewrites:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _result_from_cache(self, cached_payload: dict[str, Any], effective_filter: dict[str, Any]) -> RetrievalResult:
        """Hydrate a cached retrieval result."""
        cached_items = list(cached_payload.get("sources") or [])
        chunks = self._load_chunks_by_ids([str(item.get("chunk_id") or "") for item in cached_items])
        by_id = {chunk.id: chunk for chunk in chunks}
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
        candidate_source: str | None = None,
    ) -> RetrievalResult:
        """Build an empty retrieval result."""
        if prefilter_child_count is None:
            prefilter_child_count = self._prefilter_child_count()
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
                "vector_backend": self.vector_index.backend,
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "metadata_filtering": bool(effective_filter),
                "channels": {"dense": 0, "bm25": 0, "rrf": 0, "rerank": 0},
                "filter": effective_filter,
                "candidate_source": candidate_source or self._candidate_source(),
            },
        )

    def _empty_rewrite_result(
        self,
        original_query: str,
        query_statuses: list[dict[str, Any]],
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        coverage: dict[str, Any] | None = None,
        degraded: list[dict[str, str]] | None = None,
        candidate_source: str | None = None,
        prefilter_child_count: int | None = None,
    ) -> RetrievalResult:
        """Build an empty fused rewrite retrieval result."""
        if prefilter_child_count is None:
            prefilter_child_count = self._prefilter_child_count()
        coverage = coverage or metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        return RetrievalResult(
            sources=[],
            filter=effective_filter,
            status={
                "query": original_query,
                "query_count": len(query_statuses),
                "queries": query_statuses,
                "top_k": top_k,
                "source_count": 0,
                "prefilter_child_count": prefilter_child_count,
                "candidate_count": 0,
                "metadata_coverage": coverage,
                "cache": {"hit": False, "backend": self.cache_store.backend},
                "vector_backend": self.vector_index.backend,
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "metadata_filtering": bool(effective_filter),
                "channels": {"hit_lists": 0, "rrf": 0, "rerank_candidates": 0, "rerank": 0, "dense": 0, "bm25": 0},
                "structured_backfill": {"applied": False},
                "degraded": list(degraded or []),
                "filter": effective_filter,
                "batch_mode": True,
                "retrieval_mode": "batch_multi_query_rewrite_fused",
                "candidate_source": candidate_source or self._candidate_source(),
            },
        )

    def _cache_result(self, cache_key: str, result: RetrievalResult) -> None:
        """Write a retrieval result to cache."""
        self.cache_store.set_json(
            cache_key,
            {
                "sources": [source.payload() for source in result.sources],
                "filter": result.filter,
                "status": result.status,
            },
            ttl_seconds=self.cache_ttl_seconds,
        )
