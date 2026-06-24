"""External-index RAG engine implementation."""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.rag.engine.base import BaseRAGEngine, ChunkStore
from declaration_qa.rag.engine.common import (
    embed_chunk_batch,
    expand_parent_context,
    expand_structured_section_context,
    structured_backfill_chunk_type,
    structured_source_matches,
    unique_ordered_strings,
)
from declaration_qa.rag.model_providers import EmbeddingProvider, Reranker
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage
from declaration_qa.search.keyword import KeywordIndex
from declaration_qa.search.vector import VectorIndex
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

    def _retrieve_uncached(self, normalized_query: str, effective_filter: dict[str, Any], *, top_k: int) -> RetrievalResult:
        """Retrieve through external indexes and hydrate chunks on demand."""
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

    def _retrieve_many_uncached(
        self,
        query_specs: list[dict[str, Any]],
        results: list[RetrievalResult | None],
        effective_filter: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Batch remote dense retrieval, then hydrate and rerank."""
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
                top_k=int(spec["top_k"]),
                degraded=degraded_by_index.get(index) or [],
                batch_mode=True,
                candidate_source="external_indexes",
            )
            self._cache_result(str(spec["cache_key"]), result)
            results[index] = result
        return [result for result in results if result is not None]

    def _remote_dense_retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        *,
        limit: int,
        channel: str,
    ) -> list[RetrievedSource]:
        """Run vector retrieval against the configured vector index."""
        query_vector = self.embedding_provider.embed_query(query)
        return self._sources_from_scored_ids(
            self.vector_index.search(query_vector, filters, limit=limit),
            score_kind="dense",
            channel=channel,
        )

    def _keyword_retrieve(self, query: str, filters: dict[str, Any], *, limit: int) -> list[RetrievedSource]:
        """Run keyword retrieval against the configured keyword index."""
        return self._sources_from_scored_ids(
            self.keyword_index.search(query, filters, limit=limit),
            score_kind="bm25",
            channel=f"{self.keyword_index.backend}_bm25",
        )

    def _load_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """Load chunks from the persistent chunk store."""
        unique_ids = unique_ordered_strings(chunk_ids)
        if not unique_ids:
            return []
        chunks = self.chunk_store.get_chunks_by_ids(unique_ids) if self.chunk_store is not None else []
        by_id = {chunk.id: chunk for chunk in chunks}
        return [by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id]

    def _expand_parent_context(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        """Expand child hits with parent chunks from storage."""
        if self.chunk_store is None:
            return sources
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
        """Backfill structured section siblings from storage."""
        if self.chunk_store is None:
            return sources, {"applied": False}
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
