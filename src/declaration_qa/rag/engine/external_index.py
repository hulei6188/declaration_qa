"""外部索引 RAG 引擎实现。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.observability.langfuse import observation, record_error, safe_value
from declaration_qa.observability.performance import performance_stage
from declaration_qa.rag.engine.base import BaseRAGEngine, ChunkStore, strategy_weighted_hit_lists
from declaration_qa.rag.engine.embeddings import embed_knowledge_chunk_batch
from declaration_qa.rag.engine.parent_context import expand_parent_context
from declaration_qa.rag.engine.scoring import rrf_fuse
from declaration_qa.rag.engine.text import unique_ordered_strings
from declaration_qa.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from declaration_qa.rag.model_providers.rerankers.reranker import Reranker
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage, published_visibility_filter
from declaration_qa.search.keyword import KeywordIndex
from declaration_qa.search.vector.base import VectorIndex
from declaration_qa.storage.cache import CacheStore


MAX_PARALLEL_RERANK_WORKERS = 3


class ExternalIndexRAGEngine(BaseRAGEngine):
    """基于外部向量/关键词索引和片段存储的 RAG 引擎。"""

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
        rerank_max_sources: int = 20,
        retrieval_candidates_per_query: int = 30,
        rerank_top_n_per_query: int = 12,
    ) -> None:
        """初始化外部索引 RAG 引擎。"""
        if vector_index is None or keyword_index is None or chunk_store is None:
            raise ValueError("ExternalIndexRAGEngine requires vector_index, keyword_index and chunk_store")
        self._documents: dict[str, KnowledgeDocument] = {}
        self._staged_vectors: dict[str, dict[str, list[float]]] = {}
        super().__init__(
            cache_store=cache_store,
            vector_index=vector_index,
            embedding_provider=embedding_provider,
            reranker=reranker,
            keyword_index=keyword_index,
            cache_ttl_seconds=cache_ttl_seconds,
            rerank_score_threshold=rerank_score_threshold,
            rerank_max_sources=rerank_max_sources,
            retrieval_candidates_per_query=retrieval_candidates_per_query,
            rerank_top_n_per_query=rerank_top_n_per_query,
        )
        self.chunk_store = chunk_store

    @property
    def chunk_count(self) -> int:
        """外部片段按需回表。"""
        return 0

    def documents(self) -> list[KnowledgeDocument]:
        """返回已知文档。"""
        return list(self._documents.values())

    def index_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """将片段写入外部向量和关键词索引。"""
        self.delete_document(document.id)
        child_chunks = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        vectors = embed_knowledge_chunk_batch(self.embedding_provider, child_chunks)
        vectors_by_chunk_id = {chunk.id: vector for chunk, vector in zip(child_chunks, vectors)}
        self._staged_vectors[document.id] = vectors_by_chunk_id
        return self._write_indexed_chunks(document, chunks, vectors_by_chunk_id)

    def promote_indexed_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """复用暂存阶段已计算的向量，更新发布版本的可见性元数据。"""
        vectors_by_chunk_id = self._staged_vectors.get(document.id)
        child_ids = {chunk.id for chunk in chunks if chunk.chunk_type == "child"}
        if vectors_by_chunk_id is None or set(vectors_by_chunk_id) != child_ids:
            raise RuntimeError("Staged vectors are unavailable for knowledge revision promotion")
        return self._write_indexed_chunks(document, chunks, vectors_by_chunk_id)

    def _write_indexed_chunks(
        self,
        document: KnowledgeDocument,
        chunks: list[KnowledgeChunk],
        vectors_by_chunk_id: dict[str, list[float]],
    ) -> int:
        """使用已计算向量写入外部索引。"""
        self._documents[document.id] = document
        for chunk in chunks:
            if chunk.chunk_type == "child":
                self.vector_index.upsert(chunk, vectors_by_chunk_id[chunk.id])
        self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self._clear_cache()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """从外部索引中删除单个文档。"""
        self._documents.pop(doc_id, None)
        self._staged_vectors.pop(doc_id, None)
        self.vector_index.delete_by_doc(doc_id)
        self.keyword_index.delete_by_doc(doc_id)
        self._clear_cache()

    def rebuild_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """根据片段重建外部向量和关键词索引。"""
        self._documents = {document.id: document for document in documents}
        self.vector_index.clear()
        self.keyword_index.clear()
        child_chunks = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        vectors = embed_knowledge_chunk_batch(self.embedding_provider, child_chunks)
        for chunk, vector in zip(child_chunks, vectors):
            self.vector_index.upsert(chunk, vector)
        self.keyword_index.upsert_chunks(chunks)
        self._flush_vector_index()
        self._clear_cache()
        return len(chunks)

    def load_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """加载文档元数据，不回表片段正文。"""
        self._documents = {document.id: document for document in documents}
        self._clear_cache()
        return len(chunks)

    def _candidate_source(self) -> str:
        """返回候选来源标签。"""
        return "external_indexes"

    def _retrieve_uncached(
        self,
        normalized_query: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """通过外部索引检索，并按需回表片段。"""
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
        """批量执行远程稠密检索，然后回表并重排。"""
        search_filter = published_visibility_filter(effective_filter)
        limits = [self._recall_limit(int(spec["top_k"])) for spec in query_specs]
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        dense_scores_by_index: dict[int, list[tuple[str, float]]] = {}
        bm25_scores_by_index: dict[int, list[tuple[str, float]]] = {}
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
                output = batch_scored_id_summary(query_specs, batch_hits, empty_reason="vector_index_empty_result")
                if output.get("empty_count"):
                    record_error(
                        span,
                        stage="rag.vector_search.batch",
                        reason=str(output.get("reason") or "vector_index_empty_result"),
                        extra={"backend": self.vector_index.backend, **output},
                    )
                span.update(output=output)
            for spec, vector_hits, limit in zip(query_specs, batch_hits, limits):
                dense_scores_by_index[int(spec["index"])] = list(vector_hits[:limit])
        except Exception as exc:
            for spec in query_specs:
                degraded_by_index[int(spec["index"])].append({"stage": "dense_retrieve_batch", "error": type(exc).__name__})

        keyword_search_many = getattr(self.keyword_index, "search_many", None)
        if callable(keyword_search_many):
            try:
                bm25_scores_by_index = {
                    int(index): list(scores)
                    for index, scores in keyword_search_many(query_specs, search_filter, limits).items()
                }
            except Exception as exc:
                for spec in query_specs:
                    degraded_by_index[int(spec["index"])].append({"stage": "bm25_retrieve_batch", "error": type(exc).__name__})
                fallback_degraded: list[dict[str, str]] = []
                bm25_scores_by_index = self._keyword_retrieve_many_fallback(
                    query_specs,
                    search_filter,
                    limits,
                    fallback_degraded,
                )
                for spec in query_specs:
                    degraded_by_index[int(spec["index"])].extend(fallback_degraded)
        else:
            fallback_degraded = []
            bm25_scores_by_index = self._keyword_retrieve_many_fallback(
                query_specs,
                search_filter,
                limits,
                fallback_degraded,
            )
            for spec in query_specs:
                degraded_by_index[int(spec["index"])].extend(fallback_degraded)

        all_chunk_ids = [
            str(chunk_id)
            for score_map in (dense_scores_by_index, bm25_scores_by_index)
            for scores in score_map.values()
            for chunk_id, _ in scores
        ]
        chunks_by_id = {chunk.id: chunk for chunk in self._load_chunks_by_ids(all_chunk_ids)}
        def build_query_result(spec: dict[str, Any], limit: int) -> tuple[int, str, RetrievalResult]:
            index = int(spec["index"])
            query = str(spec["query"])
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
                retrieval_plan=spec.get("retrieval_plan") or retrieval_plan,
            )
            return index, str(spec["cache_key"]), result

        built_results: dict[int, tuple[str, RetrievalResult]] = {}
        max_workers = min(MAX_PARALLEL_RERANK_WORKERS, len(query_specs))
        if max_workers <= 1:
            index, cache_key, result = build_query_result(query_specs[0], limits[0])
            built_results[index] = (cache_key, result)
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-rerank") as executor:
                futures = {
                    executor.submit(copy_context().run, build_query_result, spec, limit): int(spec["index"])
                    for spec, limit in zip(query_specs, limits)
                }
                for future in as_completed(futures):
                    index, cache_key, result = future.result()
                    built_results[index] = (cache_key, result)

        for spec in query_specs:
            index = int(spec["index"])
            cache_key, result = built_results[index]
            self._cache_result(cache_key, result)
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
        """批量召回改写查询，全局融合后统一重排一次。"""
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
                input={"query_count": len(query_texts), "queries": query_texts, "mode": "agent_queries"},
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
                input={"query_count": len(query_specs), "limit": max(limits), "filter": search_filter, "mode": "agent_queries"},
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                batch_hits = self.vector_index.search_batch(query_vectors, search_filter, limit=max(limits))
                output = batch_scored_id_summary(query_specs, batch_hits, empty_reason="vector_index_empty_result")
                if output.get("empty_count"):
                    record_error(
                        span,
                        stage="rag.vector_search.batch",
                        reason=str(output.get("reason") or "vector_index_empty_result"),
                        extra={"backend": self.vector_index.backend, **output},
                    )
                span.update(output=output)
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
                    normalized_results = {int(index): list(scores) for index, scores in many_results.items()}
                    output = score_map_summary(query_specs, normalized_results, empty_reason="keyword_index_empty_result")
                    if output.get("empty_count"):
                        record_error(
                            span,
                            stage="rag.keyword_search.batch",
                            reason=str(output.get("reason") or "keyword_index_empty_result"),
                            extra={"backend": self.keyword_index.backend, **output},
                        )
                    span.update(output=output)
                bm25_scores_by_index.update(normalized_results)
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
            weighted_lists, channel_weights = strategy_weighted_hit_lists(
                str(spec.get("strategy") or "hybrid"),
                dense_hits,
                bm25_hits,
            )
            query_hits = rrf_fuse(weighted_lists, rrf_k=60) if weighted_lists else []
            if query_hits:
                hit_lists.append(query_hits)
            unique_chunk_ids = {source.chunk.id for source in [*dense_hits, *bm25_hits]}
            query_statuses.append(
                {
                    "id": spec.get("id") or "",
                    "strategy": spec.get("strategy") or "",
                    "aspect": spec.get("aspect") or "general",
                    "query": spec.get("query") or "",
                    "top_k": int(spec.get("top_k") or 3),
                    "source_count": len(unique_chunk_ids),
                    "channels": {"dense": len(dense_hits), "bm25": len(bm25_hits)},
                    "channel_weights": channel_weights,
                    "coverage_candidate": query_hits[0].chunk.id if query_hits else "",
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

    def _retrieve_image_uncached(
        self,
        image_path: str,
        text: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """使用用户上传图片向量检索外部索引。"""
        search_filter = published_visibility_filter(effective_filter)
        coverage = metadata_coverage(effective_filter, chain=str(effective_filter.get("chain") or KNOWLEDGE_RETRIEVAL_CHAIN))
        limit = self._recall_limit(top_k)
        degraded: list[dict[str, str]] = []
        result_query = self.image_rerank_query(text, retrieval_plan)
        try:
            with observation(
                "rag.embedding.image_query",
                input={"text": text, "image_path": image_path},
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                query_vector = self.embed_image_query(image_path, text)
                span.update(
                    output={
                        "dimension": len(query_vector),
                        "provider": self.embedding_provider.name,
                        "mode": "image_text_fusion" if text else "image",
                    }
                )
            with observation(
                "rag.vector_search.image_query",
                input={"limit": limit, "filter": search_filter},
                metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
            ) as span:
                scored_ids = self.vector_index.search(query_vector, search_filter, limit=limit)
                output = scored_id_output(scored_ids, empty_reason="image_query_vector_empty_result")
                if output.get("empty"):
                    record_error(
                        span,
                        stage="rag.vector_search.image_query",
                        reason=str(output.get("reason") or "image_query_vector_empty_result"),
                        extra={"backend": self.vector_index.backend, **output},
                    )
                span.update(output=output)
            dense_hits = self._sources_from_scored_ids(scored_ids, score_kind="dense", channel="image_dense")
        except Exception as exc:
            dense_hits = []
            degraded.append({"stage": "image_dense_retrieve", "error": type(exc).__name__})
        return self._build_result_from_hits(
            result_query,
            effective_filter,
            [source.chunk for source in dense_hits],
            len(dense_hits),
            coverage,
            dense_hits,
            [],
            top_k=top_k,
            degraded=degraded,
            candidate_source="image_query",
            retrieval_plan=retrieval_plan,
        )

    def _keyword_retrieve_many_fallback(
        self,
        query_specs: list[dict[str, Any]],
        effective_filter: dict[str, Any],
        limits: list[int],
        degraded: list[dict[str, str]],
    ) -> dict[int, list[tuple[str, float]]]:
        """批量搜索不可用时逐条执行关键词检索。"""
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
                    output = scored_id_output(results[index], empty_reason="keyword_index_empty_result")
                    if output.get("empty"):
                        record_error(
                            span,
                            stage="rag.keyword_search",
                            reason=str(output.get("reason") or "keyword_index_empty_result"),
                            extra={"backend": self.keyword_index.backend, "query_index": index, **output},
                        )
                    span.update(output=output)
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
        """在配置的向量索引上执行向量检索。"""
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
            output = scored_id_output(scored_ids, empty_reason="vector_index_empty_result")
            if output.get("empty"):
                record_error(
                    span,
                    stage="rag.vector_search",
                    reason=str(output.get("reason") or "vector_index_empty_result"),
                    extra={"backend": self.vector_index.backend, "channel": channel, **output},
                )
            span.update(output=output)
        return self._sources_from_scored_ids(scored_ids, score_kind="dense", channel=channel)

    def _keyword_retrieve(self, query: str, filters: dict[str, Any], *, limit: int) -> list[RetrievedSource]:
        """在配置的关键词索引上执行关键词检索。"""
        with observation(
            "rag.keyword_search",
            input={"query": query, "limit": limit, "filter": filters, "backend": self.keyword_index.backend},
            metadata=self._observability_metadata(),
        ) as span:
            scored_ids = self.keyword_index.search(query, filters, limit=limit)
            output = scored_id_output(scored_ids, empty_reason="keyword_index_empty_result")
            if output.get("empty"):
                record_error(
                    span,
                    stage="rag.keyword_search",
                    reason=str(output.get("reason") or "keyword_index_empty_result"),
                    extra={"backend": self.keyword_index.backend, **output},
                )
            span.update(output=output)
        return self._sources_from_scored_ids(scored_ids, score_kind="bm25", channel=f"{self.keyword_index.backend}_bm25")

    def _load_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """从持久化片段存储加载片段。"""
        unique_ids = unique_ordered_strings(chunk_ids)
        if not unique_ids:
            return []
        with performance_stage(
            "rag.chunk_hydrate",
            requested_count=len(unique_ids),
        ), observation(
            "rag.chunk_hydrate",
            input={"requested_count": len(unique_ids), "chunk_ids": unique_ids[:20]},
            metadata=self._observability_metadata(),
        ) as span:
            chunks = self.chunk_store.get_chunks_by_ids(unique_ids) if self.chunk_store is not None else []
            by_id = {chunk.id: chunk for chunk in chunks}
            ordered_chunks = [by_id[chunk_id] for chunk_id in unique_ids if chunk_id in by_id]
            missing_ids = [chunk_id for chunk_id in unique_ids if chunk_id not in by_id]
            output = {
                "loaded_count": len(ordered_chunks),
                "missing_count": len(missing_ids),
                "missing_chunk_ids": missing_ids[:20],
            }
            if not ordered_chunks:
                output["reason"] = "all_requested_chunks_missing"
            elif missing_ids:
                output["reason"] = "some_requested_chunks_missing"
            else:
                output["reason"] = "all_requested_chunks_loaded"
            if missing_ids:
                record_error(
                    span,
                    stage="rag.chunk_hydrate",
                    reason=str(output["reason"]),
                    extra=output,
                )
            span.update(output=output)
            return ordered_chunks

    def _expand_parent_context(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        """用存储中的父片段扩展子命中。"""
        if self.chunk_store is None:
            return sources
        parent_ids = [
            source.chunk.parent_id
            for source in sources
            if source.chunk.chunk_type == "child" and source.chunk.parent_id
        ]
        unique_parent_ids = unique_ordered_strings(parent_ids)
        with performance_stage(
            "rag.parent_context_load",
            parent_count=len(unique_parent_ids),
        ), observation(
            "rag.parent_context.load_parents",
            input={"parent_count": len(unique_parent_ids), "parent_ids": unique_parent_ids[:20]},
            metadata=self._observability_metadata(),
        ) as span:
            parent_chunks = self.chunk_store.get_parent_chunks(unique_parent_ids)
            parents = {chunk.id: chunk for chunk in parent_chunks}
            missing_parent_ids = [parent_id for parent_id in unique_parent_ids if parent_id not in parents]
            output = {
                "loaded_count": len(parent_chunks),
                "missing_count": len(missing_parent_ids),
                "missing_parent_ids": missing_parent_ids[:20],
            }
            if not unique_parent_ids:
                output["reason"] = "no_parent_ids"
            elif not parent_chunks:
                output["reason"] = "parents_not_found"
            elif missing_parent_ids:
                output["reason"] = "some_parents_missing"
            else:
                output["reason"] = "parents_loaded"
            if output["reason"] in {"parents_not_found", "some_parents_missing"}:
                record_error(
                    span,
                    stage="rag.parent_context.load_parents",
                    reason=str(output["reason"]),
                    extra=output,
                )
            span.update(output=output)
        return expand_parent_context(sources, parents)


def scored_id_summary(scored_ids: list[tuple[str, float]], *, limit: int = 8) -> list[dict[str, Any]]:
    """返回检索观测使用的紧凑带分数 ID 详情。"""
    return safe_value(
        [
            {"chunk_id": str(chunk_id), "score": round(float(score or 0.0), 4)}
            for chunk_id, score in (scored_ids or [])[:limit]
        ]
    )


def scored_id_output(scored_ids: list[tuple[str, float]], *, empty_reason: str) -> dict[str, Any]:
    """返回单次带分数 ID 搜索的 Langfuse 输出载荷。"""
    output: dict[str, Any] = {
        "hit_count": len(scored_ids),
        "hits": scored_id_summary(scored_ids),
    }
    if not scored_ids:
        output["empty"] = True
        output["reason"] = empty_reason
    return safe_value(output)


def batch_scored_id_summary(
    query_specs: list[dict[str, Any]],
    batch_hits: list[list[tuple[str, float]]],
    *,
    empty_reason: str,
) -> dict[str, Any]:
    """返回批量带分数 ID 搜索的 Langfuse 输出载荷。"""
    hit_counts = {
        str(int(spec["index"])): len(hits)
        for spec, hits in zip(query_specs, batch_hits)
    }
    empty_indices = [
        int(spec["index"])
        for spec, hits in zip(query_specs, batch_hits)
        if not hits
    ]
    output: dict[str, Any] = {
        "hit_counts": hit_counts,
        "empty_count": len(empty_indices),
        "empty_indices": empty_indices[:20],
    }
    if empty_indices:
        output["reason"] = empty_reason if len(empty_indices) == len(hit_counts) else "some_queries_empty"
    return safe_value(output)


def score_map_summary(
    query_specs: list[dict[str, Any]],
    scores_by_index: dict[int, list[tuple[str, float]]],
    *,
    empty_reason: str,
) -> dict[str, Any]:
    """返回索引化带分数 ID 映射的 Langfuse 输出载荷。"""
    hit_counts = {
        str(int(spec["index"])): len(scores_by_index.get(int(spec["index"])) or [])
        for spec in query_specs
    }
    empty_indices = [
        int(spec["index"])
        for spec in query_specs
        if not scores_by_index.get(int(spec["index"]))
    ]
    output: dict[str, Any] = {
        "hit_counts": hit_counts,
        "empty_count": len(empty_indices),
        "empty_indices": empty_indices[:20],
    }
    if empty_indices:
        output["reason"] = empty_reason if len(empty_indices) == len(hit_counts) else "some_queries_empty"
    return safe_value(output)
