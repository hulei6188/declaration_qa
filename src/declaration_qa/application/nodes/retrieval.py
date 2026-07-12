"""申报问答工作流的检索和证据节点。"""

from __future__ import annotations

from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.helpers.text_safety import mask_sensitive_text
from declaration_qa.application.helpers.retrieval_sources import (
    dedupe_retrieval_queries,
    merge_retrieval_sources,
    query_status_from_result,
    select_answer_sources,
)
from declaration_qa.application.helpers.knowledge_context import (
    knowledge_rag_query,
)
from declaration_qa.core.domain import RetrievalResult
from declaration_qa.observability.langfuse import observation, safe_value
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.retrieval_plan import normalize_retrieval_plan


class RetrievalNodes:
    """只暴露 Agent knowledge_search 内部使用的多路知识检索。"""

    def multi_retrieve(self, state: DeclarationQAState, filters: dict[str, Any]) -> RetrievalResult:
        """按 Agent 查询计划执行多路检索并去重合并。"""
        retrieval_queries = state.get("retrieval_queries") or []
        answer_query = knowledge_rag_query(state)
        retrieval_plan = normalize_retrieval_plan(state.get("retrieval_plan") or {})
        query_image = state.get("query_image") or {}
        query_image_path = str(query_image.get("file_path") or "").strip() if isinstance(query_image, dict) else ""
        image_only = is_image_only_query(state)
        ocr_query = query_image_ocr_query(state)
        image_query_text = "" if image_only else answer_query
        image_semantic_query = "" if image_only else answer_query
        image_retrieval_plan = {
            **retrieval_plan,
            "query_modality": "image",
            "image_rerank_query": image_semantic_query,
        }
        if query_image_path:
            image_retrieval_plan["query_image_path"] = query_image_path
        if not image_only and not retrieval_queries:
            retrieval_queries = [{"id": "rq1", "text": answer_query, "strategy": "fallback", "top_k": 4}]
        if ocr_query and not any(str(query.get("strategy") or "") == "image_ocr" for query in retrieval_queries):
            retrieval_queries = [
                *retrieval_queries,
                {"id": "image_ocr", "text": ocr_query, "strategy": "image_ocr", "top_k": 6},
            ]
        executable_queries = dedupe_retrieval_queries(retrieval_queries)
        configured_limit = int(getattr(self.rag_engine, "rerank_max_sources", 20) or 20)
        text_retrieval_enabled = bool(executable_queries)
        query_statuses: list[dict[str, Any] | None] = [None] * len(executable_queries)
        image_result: RetrievalResult | None = None
        image_query_status: dict[str, Any] | None = None
        observation_metadata = retrieval_observation_metadata(state)
        if query_image_path:
            try:
                with observation(
                    "workflow.retrieve.image_query",
                    input={
                        "query": image_semantic_query,
                        "embedding_text": image_query_text,
                        "rerank_query": image_semantic_query,
                        "image_only": image_only,
                        "text_present": not image_only,
                        "ocr_text_present": bool(ocr_query),
                        "top_k": 4,
                        "filter": filters,
                        "retrieval_plan": image_retrieval_plan,
                    },
                    metadata=observation_metadata,
                ) as span:
                    image_result = self.rag_engine.retrieve_image(
                        query_image_path,
                        image_query_text,
                        filters,
                        top_k=4,
                        retrieval_plan=image_retrieval_plan,
                    )
                    image_query_status = {
                        "id": "image",
                        "strategy": "image_query",
                        "query": image_semantic_query,
                        "embedding_text": image_query_text,
                        "rerank_query": (image_result.status or {}).get("query") or image_semantic_query,
                        "top_k": 4,
                        "source_count": len(image_result.sources),
                        "cache": (image_result.status or {}).get("cache") or {},
                        "channels": (image_result.status or {}).get("channels") or {},
                    }
                    span.update(
                        output=safe_value(
                            {
                                "source_count": len(image_result.sources),
                                "status": image_result.status or {},
                            }
                        )
                    )
            except Exception as exc:
                image_query_status = {
                    "id": "image",
                    "strategy": "image_query",
                    "query": image_semantic_query,
                    "embedding_text": image_query_text,
                    "rerank_query": image_semantic_query,
                    "top_k": 4,
                    "source_count": 0,
                    "error": type(exc).__name__,
                }

        def retrieve_one(retrieval_query: dict[str, Any]) -> tuple[dict[str, Any], RetrievalResult]:
            text = str(retrieval_query.get("text") or "").strip()
            top_k = int(retrieval_query.get("top_k") or 3)
            with observation(
                "workflow.retrieve.text_query",
                input={
                    "id": retrieval_query.get("id") or "",
                    "strategy": retrieval_query.get("strategy") or "",
                    "query": text,
                    "top_k": top_k,
                    "filter": filters,
                    "retrieval_plan": retrieval_plan,
                },
                metadata=observation_metadata,
            ) as span:
                result = self.rag_engine.retrieve(text, filters, top_k=top_k, retrieval_plan=retrieval_plan)
                span.update(
                    output=safe_value(
                        {
                            "source_count": len(result.sources),
                            "status": result.status or {},
                        }
                    )
                )
            return (
                {
                    "id": retrieval_query.get("id") or "",
                    "strategy": retrieval_query.get("strategy") or "",
                    "query": text,
                    "top_k": top_k,
                    "source_count": len(result.sources),
                    "cache": (result.status or {}).get("cache") or {},
                    "channels": (result.status or {}).get("channels") or {},
                },
                result,
            )

        batch_degraded = ""
        results: list[RetrievalResult] = []
        if not executable_queries and image_only:
            retrieval_mode = "image_only"
        elif len(executable_queries) > 1:
            try:
                with observation(
                    "workflow.retrieve.agent_queries",
                    input={
                        "query": answer_query,
                        "query_count": len(executable_queries),
                        "queries": executable_queries,
                        "top_k": configured_limit,
                        "filter": filters,
                        "retrieval_plan": retrieval_plan,
                    },
                    metadata=observation_metadata,
                ) as span:
                    result = self.rag_engine.retrieve_rewrites(
                        answer_query,
                        executable_queries,
                        filters,
                        top_k=configured_limit,
                        retrieval_plan=retrieval_plan,
                    )
                    span.update(
                        output=safe_value(
                            {
                                "source_count": len(result.sources),
                                "status": result.status or {},
                            }
                        )
                    )
                status = {
                    **(result.status or {}),
                    "original_query_count": len([query for query in retrieval_queries if query.get("text")]),
                    "deduped_query_count": len(executable_queries),
                    "retrieval_parallelism": 1,
                    "batch_degraded": "",
                }
                if image_result is None:
                    return RetrievalResult(sources=result.sources, filter=result.filter, status=status)
                results = [result]
                query_statuses = list(status.get("queries") or [])
                retrieval_mode = "multimodal_agent_query"
            except Exception as exc:
                batch_degraded = type(exc).__name__
            try:
                if not (image_result is not None and results):
                    retrieval_mode = "batch_agent_query"
                    with observation(
                        "workflow.retrieve.text_batch",
                        input={
                            "query_count": len(executable_queries),
                            "queries": executable_queries,
                            "filter": filters,
                            "retrieval_plan": retrieval_plan,
                        },
                        metadata=observation_metadata,
                    ) as span:
                        results = self.rag_engine.retrieve_many(executable_queries, filters, retrieval_plan=retrieval_plan)
                        span.update(
                            output=safe_value(
                                {
                                    "result_count": len(results),
                                    "source_count": sum(len(result.sources) for result in results),
                                    "statuses": [result.status for result in results],
                                }
                            )
                        )
                    for index, (retrieval_query, result) in enumerate(zip(executable_queries, results)):
                        query_statuses[index] = query_status_from_result(retrieval_query, result)
            except Exception as exc:
                if not results:
                    batch_degraded = batch_degraded or type(exc).__name__
                    query_statuses = [None] * len(executable_queries)
                    results = []

        if text_retrieval_enabled and not results and len(executable_queries) > 1 and self.max_retrieval_workers > 1:
            retrieval_mode = "parallel_agent_query"
            results_by_index: list[RetrievalResult | None] = [None] * len(executable_queries)
            with ThreadPoolExecutor(max_workers=min(self.max_retrieval_workers, len(executable_queries))) as executor:
                futures = {
                    executor.submit(copy_context().run, retrieve_one, retrieval_query): index
                    for index, retrieval_query in enumerate(executable_queries)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    query_status, result = future.result()
                    query_statuses[index] = query_status
                    results_by_index[index] = result
            results = [result for result in results_by_index if result is not None]
        elif text_retrieval_enabled and not results:
            retrieval_mode = "image_ocr_hybrid" if image_only else "agent_query"
            results = []
            for index, retrieval_query in enumerate(executable_queries):
                query_status, result = retrieve_one(retrieval_query)
                query_statuses[index] = query_status
                results.append(result)
        if image_result is not None:
            results.append(image_result)
        merged = merge_retrieval_sources(results)
        combined_query = "\n".join(str(query.get("text") or "") for query in executable_queries if query.get("text"))
        score_threshold = float(getattr(self.rag_engine, "rerank_score_threshold", 0.0) or 0.0)
        sources = select_answer_sources(merged, answer_query, limit=None, retrieval_plan=retrieval_plan)
        ordered_statuses = [status for status in query_statuses if status is not None]
        if image_query_status is not None:
            ordered_statuses.append(image_query_status)
        status = {
            "query": answer_query,
            "rewrite_query": combined_query,
            "query_count": len([query for query in executable_queries if query.get("text")]),
            "original_query_count": len([query for query in retrieval_queries if query.get("text")]),
            "deduped_query_count": len(executable_queries),
            "queries": ordered_statuses,
            "source_count": len(sources),
            "filter": filters,
            "metadata_filtering": bool(filters),
            "retrieval_mode": retrieval_mode,
            "query_modalities": retrieval_query_modalities(
                image_only=image_only,
                has_query_image=bool(query_image_path),
                has_ocr_text=bool(ocr_query),
            ),
            "text_retrieval_skipped": not text_retrieval_enabled,
            "text_retrieval_skip_reason": "query_image_ocr_unavailable" if image_only and not text_retrieval_enabled else "",
            "query_image_ocr": state.get("query_image_ocr_status") or {},
            "retrieval_parallelism": min(self.max_retrieval_workers, max(1, len(executable_queries))),
            "batch_degraded": batch_degraded,
            "score_cutoff": {
                "enabled": score_threshold > 0,
                "threshold": score_threshold,
                "max_sources": configured_limit,
                "max_sources_per_rerank": configured_limit,
                "merged_source_limit": None,
            },
            "retrieval_plan": retrieval_plan,
        }
        return RetrievalResult(sources=sources, filter=filters, status=status)


def retrieval_observation_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """构建检索节点观测的紧凑元数据。"""
    return safe_value(
        {
            "langfuse_trace_id": state.get("langfuse_trace_id") or "",
            "user_id": state.get("user_id"),
            "session_id": state.get("session_id") or "",
            "route": state.get("route") or "",
            "intent": state.get("intent") or "",
            "retrieval_chain": state.get("retrieval_chain") or "",
            "retrieval_plan": state.get("retrieval_plan") or {},
        }
    )


def is_image_only_query(state: dict[str, Any]) -> bool:
    """判断本次请求是否只有图片、没有用户输入文本。"""
    query_image = state.get("query_image") or {}
    if not isinstance(query_image, dict) or not query_image:
        return False
    return not bool(query_image.get("text_present", True))


def query_image_ocr_query(state: dict[str, Any]) -> str:
    """返回适合文本向量和 BM25 的查询图片 OCR 文本。"""
    return normalize_text(mask_sensitive_text(str(state.get("query_image_ocr_text") or "")))


def retrieval_query_modalities(*, image_only: bool, has_query_image: bool, has_ocr_text: bool) -> list[str]:
    """返回本次实际参与召回的查询模态。"""
    if image_only:
        return ["image", "ocr_text"] if has_ocr_text else ["image"]
    modalities = ["text"]
    if has_query_image:
        modalities.append("image")
    if has_ocr_text:
        modalities.append("ocr_text")
    return modalities
