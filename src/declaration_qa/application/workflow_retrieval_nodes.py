"""Retrieval and evidence nodes for the declaration QA workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.workflow_helpers.events import add_event
from declaration_qa.application.workflow_helpers.intent import infer_biz_type
from declaration_qa.application.workflow_helpers.retrieval_sources import (
    dedupe_retrieval_queries,
    merge_retrieval_sources,
    query_status_from_result,
    select_answer_sources,
)
from declaration_qa.application.workflow_helpers.ticket_context import (
    base_policy_filter,
    knowledge_rag_query,
    ticket_rag_query,
)
from declaration_qa.core.domain import RetrievalResult
from declaration_qa.rag.evidence import decide_evidence
from declaration_qa.rag.retrieval import KNOWLEDGE_RETRIEVAL_CHAIN
from declaration_qa.rag.router import ROUTE_TICKET_CONTEXT_RELATED


class WorkflowRetrievalMixin:
    """Knowledge retrieval and evidence-gating nodes."""
    def knowledge_filter_build(self, state: DeclarationQAState) -> DeclarationQAState:
        """构造普通知识检索过滤条件。"""
        next_state = dict(state)
        if not next_state.get("biz_type") and not next_state.get("apply_context"):
            next_state["biz_type"] = infer_biz_type(next_state.get("normalized_query") or "")
        filters = base_policy_filter(next_state, allow_request_filter=True)
        next_state["retrieval_filter"] = filters
        add_event(next_state, "knowledge_filter", filters)
        return next_state

    def query_rewrite(self, state: DeclarationQAState) -> DeclarationQAState:
        """生成多路检索 query。"""
        next_state = dict(state)
        rewrite = self.query_rewriter.rewrite(next_state)
        payload = rewrite.payload()
        next_state["query_rewrite"] = payload
        next_state["retrieval_queries"] = payload.get("queries") or []
        next_state["rag_query"] = "\n".join(str(query.get("text") or "") for query in next_state["retrieval_queries"])
        add_event(
            next_state,
            "query_rewrite",
            {
                "count": len(next_state["retrieval_queries"]),
                "retrieval_focus": payload.get("retrieval_focus"),
                "ticket_fact_type": payload.get("ticket_fact_type"),
                "provider": payload.get("provider"),
                "reason": payload.get("reason"),
                "scope_text": payload.get("scope_text") or "",
                "queries": next_state["retrieval_queries"],
            },
        )
        return next_state

    def multi_retrieve(self, state: DeclarationQAState, filters: dict[str, Any]) -> RetrievalResult:
        """按 rewrite 结果执行多路检索并去重合并。"""
        retrieval_queries = state.get("retrieval_queries") or []
        if not retrieval_queries:
            fallback_query = ticket_rag_query(state) if state.get("route") == ROUTE_TICKET_CONTEXT_RELATED else knowledge_rag_query(state)
            retrieval_queries = [{"id": "rq1", "text": fallback_query, "strategy": "fallback", "top_k": 4}]
        executable_queries = dedupe_retrieval_queries(retrieval_queries)
        query_statuses: list[dict[str, Any] | None] = [None] * len(executable_queries)

        def retrieve_one(retrieval_query: dict[str, Any]) -> tuple[dict[str, Any], RetrievalResult]:
            text = str(retrieval_query.get("text") or "").strip()
            top_k = int(retrieval_query.get("top_k") or 3)
            result = self.rag_engine.retrieve(text, filters, top_k=top_k)
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
        if len(executable_queries) > 1:
            try:
                retrieval_mode = "batch_multi_query_rewrite"
                results = self.rag_engine.retrieve_many(executable_queries, filters)
                for index, (retrieval_query, result) in enumerate(zip(executable_queries, results)):
                    query_statuses[index] = query_status_from_result(retrieval_query, result)
            except Exception as exc:
                batch_degraded = type(exc).__name__
                query_statuses = [None] * len(executable_queries)
                results = []
        else:
            results = []

        if not results and len(executable_queries) > 1 and self.max_retrieval_workers > 1:
            retrieval_mode = "parallel_multi_query_rewrite"
            results_by_index: list[RetrievalResult | None] = [None] * len(executable_queries)
            with ThreadPoolExecutor(max_workers=min(self.max_retrieval_workers, len(executable_queries))) as executor:
                futures = {executor.submit(retrieve_one, retrieval_query): index for index, retrieval_query in enumerate(executable_queries)}
                for future in as_completed(futures):
                    index = futures[future]
                    query_status, result = future.result()
                    query_statuses[index] = query_status
                    results_by_index[index] = result
            results = [result for result in results_by_index if result is not None]
        elif not results:
            retrieval_mode = "multi_query_rewrite"
            results = []
            for index, retrieval_query in enumerate(executable_queries):
                query_status, result = retrieve_one(retrieval_query)
                query_statuses[index] = query_status
                results.append(result)
        merged = merge_retrieval_sources(results)
        combined_query = "\n".join(str(query.get("text") or "") for query in executable_queries if query.get("text"))
        sources = select_answer_sources(merged, combined_query, limit=6)
        ordered_statuses = [status for status in query_statuses if status is not None]
        status = {
            "query": combined_query,
            "query_count": len([query for query in executable_queries if query.get("text")]),
            "original_query_count": len([query for query in retrieval_queries if query.get("text")]),
            "deduped_query_count": len(executable_queries),
            "queries": ordered_statuses,
            "source_count": len(sources),
            "filter": filters,
            "metadata_filtering": bool(filters),
            "retrieval_mode": retrieval_mode,
            "retrieval_parallelism": min(self.max_retrieval_workers, max(1, len(executable_queries))),
            "batch_degraded": batch_degraded,
        }
        return RetrievalResult(sources=sources, filter=filters, status=status)

    def hybrid_retrieve(self, state: DeclarationQAState) -> DeclarationQAState:
        """执行普通知识问答混合检索。"""
        next_state = dict(state)
        result = self.multi_retrieve(next_state, next_state.get("retrieval_filter") or {})
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", {**result.status, "rag_query": next_state.get("rag_query") or ""})
        return next_state

    def evidence_check(self, state: DeclarationQAState) -> DeclarationQAState:
        """执行证据门控检查。"""
        next_state = dict(state)
        if next_state.get("route") == ROUTE_TICKET_CONTEXT_RELATED and next_state.get("evidence_ok"):
            add_event(next_state, "evidence", next_state.get("evidence_decision") or {})
            return next_state
        decision = decide_evidence(
            next_state.get("retrieved_sources") or [],
            next_state.get("retrieval_filter") or {},
            intent=str(next_state.get("intent") or ""),
            chain=str(next_state.get("retrieval_chain") or KNOWLEDGE_RETRIEVAL_CHAIN),
            business_payload=next_state.get("business_snapshot") or {},
        )
        next_state["evidence_ok"] = decision.ok
        next_state["evidence_decision"] = decision.payload()
        if not decision.ok:
            next_state["refuse_reason"] = decision.reason
        add_event(next_state, "evidence", decision.payload())
        return next_state
