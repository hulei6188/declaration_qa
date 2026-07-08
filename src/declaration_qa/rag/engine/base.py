"""RAG 引擎通用契约和编排。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.observability.langfuse import observation, record_error, retrieved_source_summary, safe_value
from declaration_qa.rag.engine.chunking import split_document
from declaration_qa.rag.engine.constants import EMBEDDING_DIMS
from declaration_qa.rag.engine.scoring import rrf_fuse
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from declaration_qa.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider
from declaration_qa.rag.model_providers.rerankers.heuristic_reranker import HeuristicReranker
from declaration_qa.rag.model_providers.rerankers.reranker import Reranker
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN, metadata_coverage, normalize_retrieval_filter
from declaration_qa.rag.retrieval_plan import normalize_retrieval_plan
from declaration_qa.search.keyword import KeywordIndex
from declaration_qa.search.vector.base import VectorIndex
from declaration_qa.storage.cache import CacheStore, InMemoryCacheStore


class ChunkStore(Protocol):
    """按需检索回表使用的持久化片段存储。"""

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """按公开片段 ID 加载片段。"""
        ...

    def get_parent_chunks(self, parent_ids: list[str]) -> list[KnowledgeChunk]:
        """按公开父片段 ID 加载父片段。"""
        ...

class RAGEngine(Protocol):
    """工作流使用的通用检索和索引契约。"""

    cache_store: CacheStore
    vector_index: VectorIndex
    keyword_index: KeywordIndex | None
    chunk_store: ChunkStore | None
    embedding_provider: EmbeddingProvider
    reranker: Reranker

    @property
    def chunk_count(self) -> int:
        """返回本地加载的片段数量。"""
        ...

    def documents(self) -> list[KnowledgeDocument]:
        """返回引擎已知的文档。"""
        ...

    def document_chunks(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """将文档切分为可索引片段。"""
        ...

    def index_document(self, document: KnowledgeDocument) -> int:
        """索引单个文档。"""
        ...

    def index_chunks(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> int:
        """索引预先计算好的片段。"""
        ...

    def delete_document(self, doc_id: str) -> None:
        """从索引中删除文档。"""
        ...

    def rebuild(self, documents: list[KnowledgeDocument]) -> int:
        """根据文档重建索引。"""
        ...

    def rebuild_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """根据持久化片段重建索引。"""
        ...

    def load_indexed_data(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> int:
        """加载持久化元数据，不重写外部索引。"""
        ...

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        *,
        top_k: int = 4,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """检索单个查询的来源。"""
        ...

    def retrieve_many(
        self,
        queries: list[dict[str, Any]],
        filters: dict[str, Any],
        *,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """检索多个改写查询的来源。"""
        ...

    def retrieve_rewrites(
        self,
        original_query: str,
        retrieval_queries: list[dict[str, Any]],
        filters: dict[str, Any],
        *,
        top_k: int = 6,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """检索同一用户问题下的多个改写查询。"""
        ...

    def retrieve_image(
        self,
        image_path: str,
        text: str,
        filters: dict[str, Any],
        *,
        top_k: int = 4,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """按用户上传图片检索来源。"""
        ...


class BaseRAGEngine:
    """具体检索后端共享的 RAG 编排逻辑。"""

    def __init__(
        self,
        *,
        cache_store: CacheStore | None = None,
        vector_index: VectorIndex,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        keyword_index: KeywordIndex | None = None,
        cache_ttl_seconds: int = 600,
        rerank_score_threshold: float = 0.0,
        rerank_max_sources: int = 12,
    ) -> None:
        """初始化共享的 provider 和存储。"""
        self.cache_store = cache_store or InMemoryCacheStore()
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=EMBEDDING_DIMS)
        self.reranker = reranker or HeuristicReranker()
        self.keyword_index = keyword_index
        self.chunk_store: ChunkStore | None = None
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))
        self.rerank_score_threshold = max(0.0, min(1.0, float(rerank_score_threshold or 0.0)))
        self.rerank_max_sources = max(1, int(rerank_max_sources or 12))

    def index_document(self, document: KnowledgeDocument) -> int:
        """索引单个文档。"""
        return self.index_chunks(document, split_document(document))

    def rebuild(self, documents: list[KnowledgeDocument]) -> int:
        """根据文档重建索引。"""
        return self.rebuild_indexed_data(
            documents,
            [chunk for document in documents for chunk in split_document(document)],
        )

    def document_chunks(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        """返回文档片段。"""
        return split_document(document)

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        *,
        top_k: int = 4,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """检索单个查询的来源。"""
        normalized_query = normalize_text(query)
        effective_filter = normalize_retrieval_filter(filters)
        policy = normalize_retrieval_plan(retrieval_plan)
        cache_key = self._cache_key(normalized_query, effective_filter, top_k, policy)
        with observation(
            "rag.cache.lookup",
            input={"query": normalized_query, "filter": effective_filter, "top_k": top_k, "mode": "single"},
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            cached_payload = self.cache_store.get_json(cache_key)
            span.update(output={"hit": bool(cached_payload), "backend": self.cache_store.backend})
        if cached_payload:
            return self._result_from_cache(cached_payload, effective_filter)
        result = self._retrieve_uncached(normalized_query, effective_filter, top_k=top_k, retrieval_plan=policy)
        self._cache_result(cache_key, result)
        return result

    def retrieve_many(
        self,
        queries: list[dict[str, Any]],
        filters: dict[str, Any],
        *,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """检索多个改写查询的来源。"""
        effective_filter = normalize_retrieval_filter(filters)
        policy = normalize_retrieval_plan(retrieval_plan)
        query_specs = []
        results: list[RetrievalResult | None] = [None] * len(queries)
        for index, query in enumerate(queries):
            text = normalize_text(str(query.get("text") or ""))
            top_k = int(query.get("top_k") or 3)
            if not text:
                results[index] = self._empty_result("", effective_filter, top_k=top_k)
                continue
            cache_key = self._cache_key(text, effective_filter, top_k, policy)
            with observation(
                "rag.cache.lookup",
                input={"query": text, "filter": effective_filter, "top_k": top_k, "mode": "many", "index": index},
                metadata=self._observability_metadata(retrieval_plan=policy),
            ) as span:
                cached_payload = self.cache_store.get_json(cache_key)
                span.update(output={"hit": bool(cached_payload), "backend": self.cache_store.backend})
            if cached_payload:
                results[index] = self._result_from_cache(cached_payload, effective_filter)
            else:
                query_specs.append({"index": index, "query": text, "top_k": top_k, "cache_key": cache_key})
        if not query_specs:
            return [result for result in results if result is not None]
        return self._retrieve_many_uncached(query_specs, results, effective_filter, retrieval_plan=policy)

    def retrieve_rewrites(
        self,
        original_query: str,
        retrieval_queries: list[dict[str, Any]],
        filters: dict[str, Any],
        *,
        top_k: int = 6,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """将改写查询检索为召回列表，再统一融合并重排一次。"""
        effective_filter = normalize_retrieval_filter(filters)
        policy = normalize_retrieval_plan(retrieval_plan)
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
        cache_key = self._rewrite_cache_key(normalized_query, query_specs, effective_filter, top_k, policy)
        with observation(
            "rag.cache.lookup",
            input={
                "query": normalized_query,
                "filter": effective_filter,
                "top_k": top_k,
                "mode": "rewrites",
                "query_count": len(query_specs),
            },
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            cached_payload = self.cache_store.get_json(cache_key)
            span.update(output={"hit": bool(cached_payload), "backend": self.cache_store.backend})
        if cached_payload:
            return self._result_from_cache(cached_payload, effective_filter)
        result = self._retrieve_rewrites_uncached(
            normalized_query,
            query_specs,
            effective_filter,
            top_k=top_k,
            retrieval_plan=policy,
        )
        self._cache_result(cache_key, result)
        return result

    def retrieve_image(
        self,
        image_path: str,
        text: str,
        filters: dict[str, Any],
        *,
        top_k: int = 4,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """根据用户上传图片执行向量检索。"""
        normalized_text = normalize_text(text)
        effective_filter = normalize_retrieval_filter(filters)
        policy = normalize_retrieval_plan(retrieval_plan)
        return self._retrieve_image_uncached(
            str(image_path or ""),
            normalized_text,
            effective_filter,
            top_k=top_k,
            retrieval_plan=policy,
        )

    def embed_image_query(self, image_path: str, text: str) -> list[float]:
        """调用支持图片查询的 embedding provider。"""
        embed_image = getattr(self.embedding_provider, "embed_image_query", None)
        if callable(embed_image):
            return embed_image(image_path, text=text)
        return self.embedding_provider.embed_query(text or "图片查询")

    def image_rerank_query(self, text: str, retrieval_plan: dict[str, Any] | None) -> str:
        """返回图片召回后用于重排和证据回填的语义问题。"""
        planned = str((retrieval_plan or {}).get("image_rerank_query") or "").strip()
        return normalize_text(planned or text)

    def _call_reranker(
        self,
        query: str,
        sources: list[RetrievedSource],
        filters: dict[str, Any],
        *,
        top_n: int,
        retrieval_plan: dict[str, Any],
    ) -> list[RetrievedSource]:
        """调用重排器，兼容测试中旧签名的轻量假对象。"""
        try:
            return self.reranker.rerank(query, sources, filters, top_n=top_n, retrieval_plan=retrieval_plan)
        except TypeError as exc:
            if "retrieval_plan" not in str(exc):
                raise
            return self.reranker.rerank(query, sources, filters, top_n=top_n)

    def _reranker_model(self) -> str:
        """返回当前重排模型名称。"""
        return str(getattr(self.reranker, "model", "") or "")

    def _rerank_is_multimodal(self, retrieval_plan: dict[str, Any]) -> bool:
        """判断当前重排请求是否携带多模态输入。"""
        return bool(
            str(retrieval_plan.get("query_modality") or "").lower() == "image"
            and str(retrieval_plan.get("query_image_path") or "").strip()
        )

    def _image_candidate_count(self, sources: list[RetrievedSource]) -> int:
        """统计候选来源中的图片 chunk 数量。"""
        return sum(1 for source in sources if (source.chunk.metadata or {}).get("modality") == "image")

    def _retrieve_uncached(
        self,
        normalized_query: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """不使用缓存执行检索。"""
        raise NotImplementedError

    def _retrieve_many_uncached(
        self,
        query_specs: list[dict[str, Any]],
        results: list[RetrievalResult | None],
        effective_filter: dict[str, Any],
        *,
        retrieval_plan: dict[str, Any],
    ) -> list[RetrievalResult]:
        """不使用缓存执行批量检索。"""
        raise NotImplementedError

    def _retrieve_rewrites_uncached(
        self,
        original_query: str,
        query_specs: list[dict[str, Any]],
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """不使用缓存检索改写召回列表。"""
        raise NotImplementedError

    def _retrieve_image_uncached(
        self,
        image_path: str,
        text: str,
        effective_filter: dict[str, Any],
        *,
        top_k: int,
        retrieval_plan: dict[str, Any],
    ) -> RetrievalResult:
        """不使用缓存执行图片查询检索。"""
        raise NotImplementedError

    def _flush_vector_index(self) -> None:
        """后端支持时让新写入的向量行可见。"""
        flush = getattr(self.vector_index, "flush", None)
        if callable(flush):
            flush()

    def _candidate_source(self) -> str:
        """返回候选来源标签。"""
        return "memory"

    def _prefilter_child_count(self) -> int:
        """返回后端特定的子片段数量。"""
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
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """根据各通道命中构建标准检索结果。"""
        degraded = list(degraded or [])
        with observation(
            "rag.rrf_merge",
            input={"dense_count": len(dense_hits), "bm25_count": len(bm25_hits), "rrf_k": 60},
            metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
        ) as span:
            fused_hits = rrf_fuse([dense_hits, bm25_hits], rrf_k=60)
            span.update(output={"fused_count": len(fused_hits), "sources": retrieved_source_summary(fused_hits)})
        rerank_top_n = self._rerank_top_n(top_k)
        policy = normalize_retrieval_plan(retrieval_plan)
        with observation(
            "rag.rerank",
            input={
                "query": normalized_query,
                "candidate_count": len(fused_hits),
                "top_n": rerank_top_n,
                "provider": self.reranker.name,
                "model": self._reranker_model(),
                "query_modality": policy.get("query_modality") or "text",
                "multimodal": self._rerank_is_multimodal(policy),
                "image_candidate_count": self._image_candidate_count(fused_hits),
            },
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            try:
                reranked = self._call_reranker(
                    normalized_query,
                    fused_hits,
                    effective_filter,
                    top_n=rerank_top_n,
                    retrieval_plan=policy,
                )
                span.update(output={"reranked_count": len(reranked), "sources": retrieved_source_summary(reranked)})
            except Exception as exc:
                reranked = HeuristicReranker().rerank(normalized_query, fused_hits, effective_filter, top_n=rerank_top_n, retrieval_plan=policy)
                degraded.append({"stage": "rerank", "error": type(exc).__name__})
                record_error(
                    span,
                    exc,
                    stage="rag.rerank",
                    reason="rerank_provider_failed_fallback_to_heuristic",
                    extra={"fallback_provider": "heuristic", "candidate_count": len(fused_hits)},
                )
                span.update(
                    output={
                        "reranked_count": len(reranked),
                        "fallback_provider": "heuristic",
                        "error": type(exc).__name__,
                        "reason": "rerank_provider_failed_fallback_to_heuristic",
                        "sources": retrieved_source_summary(reranked),
                    }
                )
        with observation(
            "rag.score_cutoff",
            input={
                "reranked_count": len(reranked),
                "requested_top_k": top_k,
                "threshold": self.rerank_score_threshold,
                "max_sources": self.rerank_max_sources,
            },
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            selected_sources, score_cutoff_status = self._apply_rerank_score_cutoff(reranked, requested_top_k=top_k)
            if score_cutoff_status.get("rejected"):
                record_error(
                    span,
                    stage="rag.score_cutoff",
                    reason=str(score_cutoff_status.get("reason") or "score_cutoff_rejected_all"),
                    extra=score_cutoff_status,
                )
            span.update(
                output={
                    **score_cutoff_status,
                    "sources": retrieved_source_summary(selected_sources),
                }
            )
        with observation(
            "rag.parent_context",
            input={"source_count": len(selected_sources)},
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            expanded_sources = self._expand_parent_context(selected_sources)
            span.update(output={"before": len(selected_sources), "after": len(expanded_sources)})
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
                "rerank_model": self._reranker_model(),
                "metadata_filtering": bool(effective_filter),
                "channels": {
                    "dense": len(dense_hits),
                    "bm25": len(bm25_hits),
                    "rrf": len(fused_hits),
                    "rerank": len(reranked),
                    "selected": len(selected_sources),
                },
                "score_cutoff": score_cutoff_status,
                "retrieval_plan": policy,
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
        retrieval_plan: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """为相关改写查询构建一个融合后的检索结果。"""
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
        with observation(
            "rag.rrf_merge",
            input={
                "hit_list_count": len(hit_lists),
                "hit_counts": [len(hit_list) for hit_list in hit_lists],
                "rrf_k": 60,
                "mode": "rewrites",
            },
            metadata=self._observability_metadata(retrieval_plan=retrieval_plan),
        ) as span:
            fused_hits = rrf_fuse(hit_lists, rrf_k=60)
            span.update(output={"fused_count": len(fused_hits), "sources": retrieved_source_summary(fused_hits)})
        rerank_candidate_limit = min(len(fused_hits), max(top_k * 10, 40), 100)
        rerank_candidates = fused_hits[:rerank_candidate_limit]
        rerank_top_n = self._rerank_top_n(top_k)
        policy = normalize_retrieval_plan(retrieval_plan)
        with observation(
            "rag.rerank",
            input={
                "query": original_query,
                "candidate_count": len(rerank_candidates),
                "candidate_limit": rerank_candidate_limit,
                "top_n": rerank_top_n,
                "provider": self.reranker.name,
                "model": self._reranker_model(),
                "query_modality": policy.get("query_modality") or "text",
                "multimodal": self._rerank_is_multimodal(policy),
                "image_candidate_count": self._image_candidate_count(rerank_candidates),
                "mode": "rewrites",
            },
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            try:
                reranked = self._call_reranker(
                    original_query,
                    rerank_candidates,
                    effective_filter,
                    top_n=rerank_top_n,
                    retrieval_plan=policy,
                )
                span.update(output={"reranked_count": len(reranked), "sources": retrieved_source_summary(reranked)})
            except Exception as exc:
                reranked = HeuristicReranker().rerank(original_query, rerank_candidates, effective_filter, top_n=rerank_top_n, retrieval_plan=policy)
                degraded.append({"stage": "rerank", "error": type(exc).__name__})
                record_error(
                    span,
                    exc,
                    stage="rag.rerank",
                    reason="rerank_provider_failed_fallback_to_heuristic",
                    extra={"fallback_provider": "heuristic", "candidate_count": len(rerank_candidates), "mode": "rewrites"},
                )
                span.update(
                    output={
                        "reranked_count": len(reranked),
                        "fallback_provider": "heuristic",
                        "error": type(exc).__name__,
                        "reason": "rerank_provider_failed_fallback_to_heuristic",
                        "sources": retrieved_source_summary(reranked),
                    }
                )
        with observation(
            "rag.score_cutoff",
            input={
                "reranked_count": len(reranked),
                "requested_top_k": top_k,
                "threshold": self.rerank_score_threshold,
                "max_sources": self.rerank_max_sources,
                "mode": "rewrites",
            },
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            selected_sources, score_cutoff_status = self._apply_rerank_score_cutoff(reranked, requested_top_k=top_k)
            if score_cutoff_status.get("rejected"):
                record_error(
                    span,
                    stage="rag.score_cutoff",
                    reason=str(score_cutoff_status.get("reason") or "score_cutoff_rejected_all"),
                    extra=score_cutoff_status,
                )
            span.update(
                output={
                    **score_cutoff_status,
                    "sources": retrieved_source_summary(selected_sources),
                }
            )
        with observation(
            "rag.parent_context",
            input={"source_count": len(selected_sources), "mode": "rewrites"},
            metadata=self._observability_metadata(retrieval_plan=policy),
        ) as span:
            expanded_sources = self._expand_parent_context(selected_sources)
            span.update(output={"before": len(selected_sources), "after": len(expanded_sources)})
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
                "rerank_model": self._reranker_model(),
                "metadata_filtering": bool(effective_filter),
                "channels": {
                    "hit_lists": len(hit_lists),
                    "rrf": len(fused_hits),
                    "rerank_candidates": len(rerank_candidates),
                    "rerank": len(reranked),
                    "selected": len(selected_sources),
                    "dense": sum(int((status.get("channels") or {}).get("dense") or 0) for status in query_statuses),
                    "bm25": sum(int((status.get("channels") or {}).get("bm25") or 0) for status in query_statuses),
                },
                "score_cutoff": score_cutoff_status,
                "retrieval_plan": policy,
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
        """将带分数的片段 ID 回表为来源。"""
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

    def _recall_limit(self, top_k: int) -> int:
        """返回 RRF 和重排前的通道召回窗口。"""
        limit = max(int(top_k or 1) * 3, 8)
        if self.rerank_score_threshold > 0:
            limit = max(limit, self.rerank_max_sources * 3)
        return limit

    def _rerank_top_n(self, top_k: int) -> int:
        """返回基于分数截断前的重排输出数量。"""
        if self.rerank_score_threshold > 0:
            return max(int(top_k or 1), self.rerank_max_sources, 1)
        return max(int(top_k or 1), 1)

    def _apply_rerank_score_cutoff(
        self,
        reranked: list[RetrievedSource],
        *,
        requested_top_k: int,
    ) -> tuple[list[RetrievedSource], dict[str, Any]]:
        """按重排分数选择回答来源，而不是固定 Top-K。"""
        threshold = self.rerank_score_threshold
        if threshold <= 0:
            limit = max(int(requested_top_k or 1), 1)
            selected = reranked[:limit]
            return selected, {
                "enabled": False,
                "threshold": 0.0,
                "requested_top_k": limit,
                "max_sources": self.rerank_max_sources,
                "before": len(reranked),
                "after": len(selected),
                "rejected": False,
                "reason": "score_cutoff_disabled",
            }

        selected = [
            source
            for source in reranked
            if float(source.rerank_score or 0.0) >= threshold
        ][: self.rerank_max_sources]
        if not reranked:
            reason = "no_reranked_sources"
        elif not selected:
            reason = "all_rerank_scores_below_threshold"
        else:
            reason = "passed_threshold"
        return selected, {
            "enabled": True,
            "threshold": threshold,
            "requested_top_k": max(int(requested_top_k or 1), 1),
            "max_sources": self.rerank_max_sources,
            "before": len(reranked),
            "after": len(selected),
            "rejected": bool(reranked and not selected),
            "reason": reason,
        }

    def _sources_from_scored_ids_lookup(
        self,
        scored_ids: list[tuple[str, float]],
        chunks_by_id: dict[str, KnowledgeChunk],
        *,
        score_kind: str,
        channel: str,
    ) -> list[RetrievedSource]:
        """通过预加载片段映射回表带分数的 ID。"""
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
        """按公开 ID 加载片段。"""
        raise NotImplementedError

    def _unique_hit_chunks(self, *hit_lists: list[RetrievedSource]) -> list[KnowledgeChunk]:
        """返回去重后的命中片段。"""
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
        """扩展父级上下文。"""
        raise NotImplementedError

    def _cache_key(
        self,
        normalized_query: str,
        effective_filter: dict[str, Any],
        top_k: int,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> str:
        """构建检索缓存键。"""
        payload = json.dumps(
            {
                "query": normalized_query,
                "filters": effective_filter,
                "top_k": top_k,
                "retrieval_plan": normalize_retrieval_plan(retrieval_plan),
                "rerank_score_threshold": self.rerank_score_threshold,
                "rerank_max_sources": self.rerank_max_sources,
            },
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
        retrieval_plan: dict[str, Any] | None = None,
    ) -> str:
        """构建融合改写检索缓存键。"""
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
                "retrieval_plan": normalize_retrieval_plan(retrieval_plan),
                "rerank_score_threshold": self.rerank_score_threshold,
                "rerank_max_sources": self.rerank_max_sources,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return f"declaration_qa:rag:rewrites:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _result_from_cache(self, cached_payload: dict[str, Any], effective_filter: dict[str, Any]) -> RetrievalResult:
        """回表缓存中的检索结果。"""
        cached_items = list(cached_payload.get("sources") or [])
        with observation(
            "rag.cache.hydrate",
            input={"cached_source_count": len(cached_items)},
            metadata=self._observability_metadata(),
        ) as span:
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
            span.update(output={"source_count": len(cached_sources), "sources": retrieved_source_summary(cached_sources)})
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
        """构建空检索结果。"""
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
                "rerank_model": self._reranker_model(),
                "metadata_filtering": bool(effective_filter),
                "channels": {"dense": 0, "bm25": 0, "rrf": 0, "rerank": 0},
                "score_cutoff": self._empty_score_cutoff_status(top_k),
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
        """构建空的融合改写检索结果。"""
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
                "rerank_model": self._reranker_model(),
                "metadata_filtering": bool(effective_filter),
                "channels": {"hit_lists": 0, "rrf": 0, "rerank_candidates": 0, "rerank": 0, "dense": 0, "bm25": 0},
                "score_cutoff": self._empty_score_cutoff_status(top_k),
                "degraded": list(degraded or []),
                "filter": effective_filter,
                "batch_mode": True,
                "retrieval_mode": "batch_multi_query_rewrite_fused",
                "candidate_source": candidate_source or self._candidate_source(),
            },
        )

    def _cache_result(self, cache_key: str, result: RetrievalResult) -> None:
        """将检索结果写入缓存。"""
        with observation(
            "rag.cache.write",
            input={"source_count": len(result.sources), "ttl_seconds": self.cache_ttl_seconds},
            metadata=self._observability_metadata(retrieval_plan=result.status.get("retrieval_plan") or {}),
        ) as span:
            self.cache_store.set_json(
                cache_key,
                {
                    "sources": [source.payload() for source in result.sources],
                    "filter": result.filter,
                    "status": result.status,
                },
                ttl_seconds=self.cache_ttl_seconds,
            )
            span.update(output={"backend": self.cache_store.backend, "written": True})

    def _empty_score_cutoff_status(self, top_k: int) -> dict[str, Any]:
        """返回空检索结果的分数截断状态。"""
        return {
            "enabled": self.rerank_score_threshold > 0,
            "threshold": self.rerank_score_threshold,
            "requested_top_k": max(int(top_k or 1), 1),
            "max_sources": self.rerank_max_sources,
            "before": 0,
            "after": 0,
            "rejected": False,
            "reason": "empty_retrieval_result",
        }

    def _observability_metadata(self, *, retrieval_plan: dict[str, Any] | None = None) -> dict[str, Any]:
        """构建 RAG 引擎观测的紧凑元数据。"""
        return safe_value(
            {
                "rag_engine": self.__class__.__name__,
                "cache_backend": self.cache_store.backend,
                "vector_backend": self.vector_index.backend,
                "keyword_backend": self.keyword_index.backend if self.keyword_index else "",
                "embedding_provider": self.embedding_provider.name,
                "rerank_provider": self.reranker.name,
                "rerank_model": self._reranker_model(),
                "rerank_score_threshold": self.rerank_score_threshold,
                "rerank_max_sources": self.rerank_max_sources,
                "retrieval_plan": normalize_retrieval_plan(retrieval_plan),
            }
        )
