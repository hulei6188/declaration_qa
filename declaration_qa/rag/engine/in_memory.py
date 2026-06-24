"""In-memory RAG engine implementation."""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.rag.engine.base import BaseRAGEngine
from declaration_qa.rag.engine.common import (
    bm25_retrieve,
    dense_retrieve,
    embed_chunk_batch,
    expand_parent_context,
    expand_structured_section_context,
    matches_filter,
    unique_ordered_strings,
)
from declaration_qa.rag.model_providers import EmbeddingProvider, Reranker
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage
from declaration_qa.search.keyword import KeywordIndex
from declaration_qa.search.vector import MemoryVectorIndex, VectorIndex
from declaration_qa.storage.cache import CacheStore


class InMemoryRAGEngine(BaseRAGEngine):
    """RAG engine backed by process-local documents, chunks and vectors."""

    def __init__(
        self,
        *,
        cache_store: CacheStore | None = None,
        vector_index: VectorIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        keyword_index: KeywordIndex | None = None,
        cache_ttl_seconds: int = 600,
    ) -> None:
        """Initialize in-memory RAG engine."""
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
        )

    @property
    def chunk_count(self) -> int:
        """Return loaded chunk count."""
        return len(self._chunks)

    def documents(self) -> list[KnowledgeDocument]:
        """Return loaded documents."""
        return list(self._documents.values())

    def index_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """Index chunks into local memory and local indexes."""
        self.delete_document(document.id)
        self._documents[document.id] = document
        vectors = embed_chunk_batch(self.embedding_provider, [chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.embedding_id] = vector
            if chunk.chunk_type == "child":
                self.vector_index.upsert(chunk, vector)
        if self.keyword_index:
            self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self.cache_store.clear()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """Delete one document from memory and indexes."""
        self._documents.pop(doc_id, None)
        for chunk_id, chunk in list(self._chunks.items()):
            if chunk.doc_id == doc_id:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk.embedding_id, None)
        self.vector_index.delete_by_doc(doc_id)
        if self.keyword_index:
            self.keyword_index.delete_by_doc(doc_id)
        self.cache_store.clear()

    def rebuild_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """Rebuild local memory/vector/keyword indexes from chunks."""
        self._documents.clear()
        self._chunks.clear()
        self._vectors.clear()
        self.vector_index.clear()
        if self.keyword_index:
            self.keyword_index.clear()
        self._documents = {document.id: document for document in documents}
        vectors = embed_chunk_batch(self.embedding_provider, [chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.embedding_id] = vector
            if chunk.chunk_type == "child":
                self.vector_index.upsert(chunk, vector)
        if self.keyword_index:
            self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self.cache_store.clear()
        return len(chunks)

    def load_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """Load persisted document/chunk metadata without re-embedding."""
        self._documents = {document.id: document for document in documents}
        self._chunks = {chunk.id: chunk for chunk in chunks}
        self._vectors.clear()
        self.cache_store.clear()
        return len(chunks)

    def _retrieve_uncached(self, normalized_query: str, effective_filter: dict[str, Any], *, top_k: int) -> RetrievalResult:
        """Retrieve from in-memory chunks."""
        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, effective_filter)]
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
        limit = max(top_k * 3, 8)
        degraded: list[dict[str, str]] = []
        try:
            dense_hits = self._dense_retrieve(normalized_query, candidates, effective_filter, limit=limit)
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
        )

    def _retrieve_many_uncached(
        self,
        query_specs: list[dict[str, Any]],
        results: list[RetrievalResult | None],
        effective_filter: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Batch retrieve from in-memory chunks."""
        candidates = [chunk for chunk in self._chunks.values() if chunk.chunk_type == "child" and matches_filter(chunk, effective_filter)]
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

        limits = [max(int(spec["top_k"]) * 3, 8) for spec in query_specs]
        dense_by_index: dict[int, list[RetrievedSource]] = {}
        degraded_by_index: dict[int, list[dict[str, str]]] = {int(spec["index"]): [] for spec in query_specs}
        try:
            query_vectors = self.embedding_provider.embed_documents([str(spec["query"]) for spec in query_specs])
            batch_hits = self.vector_index.search_batch(query_vectors, effective_filter, limit=max(limits))
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
            )
            self._cache_result(str(spec["cache_key"]), result)
            results[index] = result
        return [result for result in results if result is not None]

    def _prefilter_child_count(self) -> int:
        """Return loaded child count."""
        return sum(1 for chunk in self._chunks.values() if chunk.chunk_type == "child")

    def _load_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """Load chunks from memory."""
        unique_ids = unique_ordered_strings(chunk_ids)
        chunks = [self._chunks[chunk_id] for chunk_id in unique_ids if chunk_id in self._chunks]
        by_id = {chunk.id: chunk for chunk in chunks}
        return [by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id]

    def _expand_parent_context(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        """Expand child hits with in-memory parents."""
        return expand_parent_context(sources, self._chunks)

    def _expand_structured_section_context(
        self,
        query: str,
        sources: list[RetrievedSource],
    ) -> tuple[list[RetrievedSource], dict[str, Any]]:
        """Backfill structured section siblings from memory."""
        return expand_structured_section_context(query, sources, self._chunks)

    def _dense_retrieve(
        self,
        query: str,
        candidates: list[KnowledgeChunk],
        filters: dict[str, Any],
        *,
        limit: int,
    ) -> list[RetrievedSource]:
        """Run vector retrieval for this engine."""
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
