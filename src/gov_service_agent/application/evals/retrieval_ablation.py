"""冻结Agent检索Query后的检索组件消融评测。"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from gov_service_agent.application.evals.evaluators import evaluate_retrieval
from gov_service_agent.rag.engine.scoring import rrf_fuse
from gov_service_agent.rag.engine.text import normalize_text


ABLATION_VARIANTS = ("bm25", "vector", "rrf", "rrf_rerank")


def extract_agent_retrieval_queries(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """从真实工具轨迹提取Agent实际执行过的知识检索Query。"""
    collected: list[dict[str, Any]] = []
    for call in trajectory.get("calls") or []:
        if str(call.get("tool_name") or "") != "knowledge_search":
            continue
        result = call.get("result") or {}
        raw_queries = result.get("executed_queries") or (call.get("arguments") or {}).get("queries") or []
        for raw in raw_queries:
            if not isinstance(raw, dict):
                continue
            text = normalize_text(str(raw.get("text") or raw.get("query") or ""))
            if not text:
                continue
            collected.append(
                {
                    "text": text,
                    "strategy": str(raw.get("strategy") or "hybrid"),
                    "aspect": str(raw.get("aspect") or "general"),
                }
            )
    return dedupe_queries(collected)


def extract_retrieval_filter(trajectory: dict[str, Any]) -> dict[str, Any]:
    """从knowledge_search结果中提取当时实际使用的业务过滤条件。"""
    for call in trajectory.get("calls") or []:
        if str(call.get("tool_name") or "") != "knowledge_search":
            continue
        retrieval_filter = (call.get("result") or {}).get("retrieval_filter")
        if isinstance(retrieval_filter, dict):
            return dict(retrieval_filter)
    return {}


def normalize_frozen_queries(value: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    queries = []
    for raw in value or []:
        if not isinstance(raw, dict):
            continue
        text = normalize_text(str(raw.get("text") or raw.get("query") or ""))
        if text:
            queries.append(
                {
                    "text": text,
                    "strategy": str(raw.get("strategy") or "hybrid"),
                    "aspect": str(raw.get("aspect") or "general"),
                }
            )
    return dedupe_queries(queries)


def run_retrieval_ablation(
    rag_engine,
    *,
    queries: list[dict[str, Any]],
    filters: dict[str, Any],
    expected_doc_ids: list[str],
    expected_chunk_ids: list[str],
    top_k: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对同一批冻结Query运行四组检索排名并计算固定多K指标。"""
    frozen = normalize_frozen_queries(queries)
    if not frozen:
        return {"status": "not_applicable", "reason": "agent_retrieval_queries_unavailable"}, []
    started = perf_counter()
    per_query = []
    rankings: dict[str, list[list[Any]]] = {name: [] for name in ABLATION_VARIANTS}
    for query in frozen:
        result = rag_engine.retrieve_variants(
            query["text"],
            filters,
            top_k=max(10, int(top_k or 10)),
            retrieval_plan={"retrieval_strategy": query.get("strategy") or "hybrid"},
        )
        for name in ABLATION_VARIANTS:
            ranking = list((result.get("variants") or {}).get(name) or [])
            if ranking:
                rankings[name].append(ranking)
        per_query.append(
            {
                "query": query,
                "status": dict(result.get("status") or {}),
            }
        )

    variant_results: dict[str, Any] = {}
    score_records: list[dict[str, Any]] = []
    for name in ABLATION_VARIANTS:
        hit_lists = rankings[name]
        merged = rrf_fuse(hit_lists, rrf_k=60) if len(hit_lists) > 1 else (hit_lists[0] if hit_lists else [])
        payloads = [source.payload() for source in merged[: max(10, int(top_k or 10))]]
        metrics, _, records = evaluate_retrieval(
            payloads,
            expected_doc_ids=expected_doc_ids,
            expected_chunk_ids=expected_chunk_ids,
            top_k=max(10, int(top_k or 10)),
        )
        fixed_metrics = {
            key: value
            for key, value in metrics.items()
            if key in {"retrieval_hit_at_5", "retrieval_recall_at_5", "retrieval_hit_at_10", "retrieval_recall_at_10", "retrieval_mrr_at_10"}
        }
        variant_results[name] = {
            "metrics": fixed_metrics,
            "retrieved_doc_ids": unique_strings(item.get("doc_id") for item in payloads[:10]),
            "retrieved_chunk_ids": unique_strings(item.get("chunk_id") for item in payloads[:10]),
        }
        for record in records:
            if record.get("metric_name") not in fixed_metrics:
                continue
            score_records.append(
                {
                    **record,
                    "metric_name": f"ablation.{name}.{record['metric_name']}",
                    "evaluator_type": "retrieval_ablation",
                    "evaluator_version": "frozen-query-ablation-v1",
                    "raw_result": {"variant": name, "query_count": len(frozen)},
                }
            )
    return {
        "status": "success",
        "query_source": "frozen_agent_queries",
        "queries": frozen,
        "filter": filters,
        "variants": variant_results,
        "per_query": per_query,
        "elapsed_ms": int((perf_counter() - started) * 1000),
    }, score_records


def dedupe_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for query in queries:
        key = (str(query.get("text") or ""), str(query.get("strategy") or "hybrid"))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        output.append(query)
    return output


def unique_strings(values) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
