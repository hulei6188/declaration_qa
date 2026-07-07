"""External-index RAG engine implementation."""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.observability.langfuse import observation, safe_value
from declaration_qa.rag.engine.base import BaseRAGEngine, ChunkStore
from declaration_qa.rag.engine.embeddings import embed_chunk_batch
from declaration_qa.rag.engine.structured_context import (
    expand_parent_context,
    expand_structured_section_context,
    structured_backfill_chunk_types,
    structured_source_matches,
)
from declaration_qa.rag.engine.text import unique_ordered_strings
from declaration_qa.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from declaration_qa.rag.model_providers.rerankers.reranker import Reranker
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage, published_visibility_filter
from declaration_qa.search.keyword import KeywordIndex
from declaration_qa.search.vector.base import VectorIndex
from declaration_qa.storage.cache import CacheStore


class ExternalIndexRAGEngine(BaseRAGEngine):
    """RAG engine backed by external vector/keyword indexes and a chunk store."""

    def __init__(
        self,
        *,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        chunk_store: ChunkStore,
        cache_store: CacheStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        cache_ttl_seconds: int = 600,
        rerank_score_threshold: float = 0.0,
        rerank_max_sources: int = 12,
    ) -> None:
        """Initialize external-index RAG engine."""
        if vector_index is None or keyword_index is None or chunk_store is None:
            raise ValueError("ExternalIndexRAGEngine requires vector_index, keyword_index and chunk_store")
        self._documents: dict[str, KnowledgeDocument] = {}
        super().__init__(
            cache_store=cache_store,
            vector_index=vector_index,
            embedding_provider=embedding_provider,
            reranker=reranker,
            keyword_index=keyword_index,
            cache_ttl_seconds=cache_ttl_seconds,
            rerank_score_threshold=rerank_score_threshold,
            rerank_max_sources=rerank_max_sources,
        )
        self.chunk_store = chunk_store

    @property
    def chunk_count(self) -> int:
        """External chunks are hydrated on demand."""
        return 0

    def documents(self) -> list[KnowledgeDocument]:
        """Return known documents."""
        return list(self._documents.values())

    def index_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """Index chunks into external vector and keyword indexes."""
        self.delete_document(document.id)
        self._documents[document.id] = document
        vectors = embed_chunk_batch(self.embedding_provider, [chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_type == "child":
                self.vector_index.upsert(chunk, vector)
        self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self.cache_store.clear()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """Delete one document from external indexes."""
        self._documents.pop(doc_id, None)
        self.vector_index.delete_by_doc(doc_id)
        self.keyword_index.delete_by_doc(doc_id)
        self.cache_store.clear()

    def rebuild_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """Rebuild external vector and keyword indexes from chunks."""
        self._documents = {document.id: document for document in documents}
        self.vector_index.clear()
        self.keyword_index.clear()
        vectors = embed_chunk_batch(self.embedding_provider, [chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_type == "child":
                self.vector_index.upsert(chunk, vector)
        self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self.cache_store.clear()
        return len(chunks)

    def load_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """Load document metadata without hydrating chunk bodies."""
        self._documents = {document.id: document for document in documents}
        self.cache_store.clear()
        return len(chunks)

    def _candidate_source(self) -> str:
        """Return candidate source label."""
        return "external_indexes"

    def _retrieve_uncached(
        self,
        normalized_query: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """Retrieve through external indexes and hydrate chunks on demand."""
        search_filter = published_visibility_filter(effective_filter)
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        limit = self._recall_limit(top_k)
        degraded: list[dict[str, str]] = []
        try:
            dense_hits = self._remote_dense_retrieve(normalized_query, search_filter, limit=limit, channel="dense")
        except Exception as exc:
            dense_hits = []
            degraded.append({"stage": "dense_retrieve", "error": type(exc).__name__})
        try:
            bm25_hits = self._keyword_retrieve(normalized_query, search_filter, limit=limit)
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
        """Batch remote dense retrieval, then hydrate and rerank."""
        search_filter = published_visibility_filter(effective_filter)
        limits = [self._recall_limit(int(spec["top_k"])) for spec in query_specs]
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        dense_by_index: dict[int, list[RetrievedSource]] = {}
        degraded_by_index: dict[int, list[dict[str, str]]] = {int(spec["index"]): [] for spec in query_specs}
        try:
            query_texts = [str(spec["query"]) for spec in query_specs]
            with observation(
                "rag.embedding.batch",
                input={"query_count": len(query_texts), "queries": query_texts},
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                query_vectors = self.embedding_provider.embed_documents(query_texts)
                span.update(
                    output={
                        "vector_count": len(query_vectors),
                        "dimension": len(query_vectors[0]) if query_vectors else 0,
                        "provider": self.embedding_provider.name,
                    }
                )
            with observation(
                "rag.vector_search.batch",
                input={"query_count": len(query_specs), "limit": max(limits), "filter": search_filter},
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                batch_hits = self.vector_index.search_batch(query_vectors, search_filter, limit=max(limits))
                span.update(output={"hit_counts": [len(hits) for hits in batch_hits]})
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
            dense_hits = dense_by_index.get(index) or []
            try:
                bm25_hits = self._keyword_retrieve(query, search_filter, limit=limit)
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
                top_k=int(spec["top_k"]),
                degraded=degraded_by_index.get(index) or [],
                batch_mode=True,
                candidate_source="external_indexes",
                retrieval_plan=retrieval_plan,
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
        """Batch recall rewrite queries, globally fuse, and rerank once."""
        search_filter = published_visibility_filter(effective_filter)
        limits = [self._recall_limit(int(spec["top_k"])) for spec in query_specs]
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        degraded: list[dict[str, str]] = []
        dense_scores_by_index: dict[int, list[tuple[str, float]]] = {int(spec["index"]): [] for spec in query_specs}
        bm25_scores_by_index: dict[int, list[tuple[str, float]]] = {int(spec["index"]): [] for spec in query_specs}

        try:
            query_texts = [str(spec["query"]) for spec in query_specs]
            with observation(
                "rag.embedding.batch",
                input={"query_count": len(query_texts), "queries": query_texts, "mode": "rewrites"},
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                query_vectors = self.embedding_provider.embed_documents(query_texts)
                span.update(
                    output={
                        "vector_count": len(query_vectors),
                        "dimension": len(query_vectors[0]) if query_vectors else 0,
                        "provider": self.embedding_provider.name,
                    }
                )
            with observation(
                "rag.vector_search.batch",
                input={"query_count": len(query_specs), "limit": max(limits), "filter": search_filter, "mode": "rewrites"},
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                batch_hits = self.vector_index.search_batch(query_vectors, search_filter, limit=max(limits))
                span.update(output={"hit_counts": [len(hits) for hits in batch_hits]})
            for spec, vector_hits, limit in zip(query_specs, batch_hits, limits):
                dense_scores_by_index[int(spec["index"])] = [
                    (str(chunk_id), float(score))
                    for chunk_id, score in vector_hits[:limit]
                    if str(chunk_id) and score > 0
                ]
        except Exception as exc:
            degraded.append({"stage": "dense_retrieve_batch", "error": type(exc).__name__})

        keyword_search_many = getattr(self.keyword_index, "search_many", None)
        if callable(keyword_search_many):
            try:
                with observation(
                    "rag.keyword_search.batch",
                    input={
                        "query_count": len(query_specs),
                        "queries": [str(spec.get("query") or "") for spec in query_specs],
                        "limits": limits,
                        "filter": search_filter,
                        "backend": self.keyword_index.backend,
                    },
                    metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
                ) as span:
                    many_results = keyword_search_many(query_specs, search_filter, limits)
                    span.update(output={"hit_counts": {str(index): len(scores) for index, scores in many_results.items()}})
                bm25_scores_by_index.update({int(index): list(scores) for index, scores in many_results.items()})
            except Exception as exc:
                degraded.append({"stage": "bm25_retrieve_batch", "error": type(exc).__name__})
                bm25_scores_by_index.update(self._keyword_retrieve_many_fallback(query_specs, search_filter, limits, degraded))
        else:
            bm25_scores_by_index.update(self._keyword_retrieve_many_fallback(query_specs, search_filter, limits, degraded))

        all_chunk_ids = [
            chunk_id
            for score_lists in (dense_scores_by_index, bm25_scores_by_index)
            for scores in score_lists.values()
            for chunk_id, _ in scores
        ]
        chunks_by_id = {chunk.id: chunk for chunk in self._load_chunks_by_ids(all_chunk_ids)}

        hit_lists: list[list[RetrievedSource]] = []
        query_statuses: list[dict[str, Any]] = []
        for spec in query_specs:
            index = int(spec["index"])
            dense_hits = self._sources_from_scored_ids_lookup(
                dense_scores_by_index.get(index) or [],
                chunks_by_id,
                score_kind="dense",
                channel=f"dense_batch:q{index}",
            )
            bm25_hits = self._sources_from_scored_ids_lookup(
                bm25_scores_by_index.get(index) or [],
                chunks_by_id,
                score_kind="bm25",
                channel=f"{self.keyword_index.backend}_bm25_batch:q{index}",
            )
            if dense_hits:
                hit_lists.append(dense_hits)
            if bm25_hits:
                hit_lists.append(bm25_hits)
            unique_chunk_ids = {source.chunk.id for source in [*dense_hits, *bm25_hits]}
            query_statuses.append(
                {
                    "id": spec.get("id") or "",
                    "strategy": spec.get("strategy") or "",
                    "query": spec.get("query") or "",
                    "top_k": int(spec.get("top_k") or 3),
                    "source_count": len(unique_chunk_ids),
                    "channels": {"dense": len(dense_hits), "bm25": len(bm25_hits)},
                }
            )

        return self._build_rewrite_result_from_hit_lists(
            original_query,
            effective_filter,
            query_specs,
            query_statuses,
            hit_lists,
            coverage,
            top_k=top_k,
            degraded=degraded,
            candidate_count=len(chunks_by_id),
            prefilter_child_count=len(chunks_by_id),
            candidate_source="external_indexes",
            retrieval_plan=retrieval_plan,
        )

    def _keyword_retrieve_many_fallback(
        self,
        query_specs: list[dict[str, Any]],
        effective_filter: dict[str, Any],
        limits: list[int],
        degraded: list[dict[str, str]],
    ) -> dict[int, list[tuple[str, float]]]:
        """Run keyword retrieval one query at a time when batch search is unavailable."""
        results: dict[int, list[tuple[str, float]]] = {}
        for spec, limit in zip(query_specs, limits):
            index = int(spec["index"])
            try:
                query = str(spec["query"])
                with observation(
                    "rag.keyword_search",
                    input={
                        "query": query,
                        "limit": limit,
                        "filter": effective_filter,
                        "backend": self.keyword_index.backend,
                        "mode": "fallback",
                    },
                    metadata=self._observability_metadata(),
                ) as span:
                    results[index] = self.keyword_index.search(query, effective_filter, limit=limit)
                    span.update(output={"hit_count": len(results[index]), "hits": scored_id_summary(results[index])})
            except Exception as exc:
                results[index] = []
                degraded.append({"stage": "bm25_retrieve", "error": type(exc).__name__})
        return results

    def _remote_dense_retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        *,
        limit: int,
        channel: str,
    ) -> list[RetrievedSource]:
        """Run vector retrieval against the configured vector index."""
        with observation(
            "rag.embedding.query",
            input={"query": query},
            metadata=self._observability_metadata(),
        ) as span:
            query_vector = self.embedding_provider.embed_query(query)
            span.update(output={"dimension": len(query_vector), "provider": self.embedding_provider.name})
        with observation(
            "rag.vector_search",
            input={"limit": limit, "filter": filters, "channel": channel},
            metadata=self._observability_metadata(),
        ) as span:
            scored_ids = self.vector_index.search(query_vector, filters, limit=limit)
            span.update(output={"hit_count": len(scored_ids), "hits": scored_id_summary(scored_ids)})
        return self._sources_from_scored_ids(scored_ids, score_kind="dense", channel=channel)

    def _keyword_retrieve(self, query: str, filters: dict[str, Any], *, limit: int) -> list[RetrievedSource]:
        """Run keyword retrieval against the configured keyword index."""
        with observation(
            "rag.keyword_search",
            input={"query": query, "limit": limit, "filter": filters, "backend": self.keyword_index.backend},
            metadata=self._observability_metadata(),
        ) as span:
            scored_ids = self.keyword_index.search(query, filters, limit=limit)
            span.update(output={"hit_count": len(scored_ids), "hits": scored_id_summary(scored_ids)})
        return self._sources_from_scored_ids(scored_ids, score_kind="bm25", channel=f"{self.keyword_index.backend}_bm25")

    def _load_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """Load chunks from the persistent chunk store."""
        unique_ids = unique_ordered_strings(chunk_ids)
        if not unique_ids:
            return []
        with observation(
            "rag.chunk_hydrate",
            input={"requested_count": len(unique_ids), "chunk_ids": unique_ids[:20]},
            metadata=self._observability_metadata(),
        ) as span:
            chunks = self.chunk_store.get_chunks_by_ids(unique_ids) if self.chunk_store is not None else []
            by_id = {chunk.id: chunk for chunk in chunks}
            ordered_chunks = [by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id]
            span.update(output={"loaded_count": len(ordered_chunks), "missing_count": len(unique_ids) - len(ordered_chunks)})
            return ordered_chunks

    def _expand_parent_context(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        """Expand child hits with parent chunks from storage."""
        if self.chunk_store is None:
            return sources
        parent_ids = [
            source.chunk.parent_id
            for source in sources
            if source.chunk.chunk_type == "child" and source.chunk.parent_id
        ]
        unique_parent_ids = unique_ordered_strings(parent_ids)
        with observation(
            "rag.parent_context.load_parents",
            input={"parent_count": len(unique_parent_ids), "parent_ids": unique_parent_ids[:20]},
            metadata=self._observability_metadata(),
        ) as span:
            parent_chunks = self.chunk_store.get_parent_chunks(unique_parent_ids)
            span.update(output={"loaded_count": len(parent_chunks)})
        parents = {chunk.id: chunk for chunk in parent_chunks}
        return expand_parent_context(sources, parents)

    def _expand_structured_section_context(
        self,
        query: str,
        sources: list[RetrievedSource],
        *,
        retrieval_plan: dict[str, Any],
    ) -> tuple[list[RetrievedSource], dict[str, Any]]:
        """Backfill structured section siblings from storage."""
        if self.chunk_store is None:
            return sources, {"applied": False}
        target_chunk_types = structured_backfill_chunk_types(query, retrieval_plan)
        if not target_chunk_types:
            return sources, {"applied": False}
        sibling_lookup = {}
        for target_chunk_type in target_chunk_types:
            parent_section_ids = [
                str(source.chunk.metadata.get("parent_section_id") or "")
                for source in sources
                if structured_source_matches(source, target_chunk_type)
            ]
            unique_parent_section_ids = unique_ordered_strings(parent_section_ids)
            with observation(
                "rag.structured_backfill.load_siblings",
                input={
                    "target_chunk_type": target_chunk_type,
                    "parent_section_count": len(unique_parent_section_ids),
                    "parent_section_ids": unique_parent_section_ids[:20],
                },
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                sibling_chunks = self.chunk_store.get_structured_section_chunks(
                    unique_parent_section_ids,
                    target_chunk_type,
                )
                span.update(output={"sibling_count": len(sibling_chunks)})
            sibling_lookup.update({chunk.id: chunk for chunk in sibling_chunks})
        return expand_structured_section_context(
            query,
            sources,
            sibling_lookup,
            retrieval_plan=retrieval_plan,
        )


def scored_id_summary(scored_ids: list[tuple[str, float]], *, limit: int = 8) -> list[dict[str, Any]]:
    """Return compact scored id details for retrieval observations."""
    return safe_value(
        [
            {"chunk_id": str(chunk_id), "score": round(float(score or 0.0), 4)}
            for chunk_id, score in (scored_ids or [])[:limit]
        ]
    )
