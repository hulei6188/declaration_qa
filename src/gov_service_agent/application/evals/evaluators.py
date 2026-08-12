"""检索、回答、结果和性能评分器。"""

from __future__ import annotations

from math import log2
import re
from typing import Any

from gov_service_agent.application.evals.trajectory import score_record


def evaluate_retrieval(
    sources: list[dict[str, Any]],
    *,
    expected_doc_ids: list[str],
    expected_chunk_ids: list[str],
    top_k: int,
    fixed_ks: tuple[int, ...] = (5, 10),
    mrr_k: int = 10,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    selected = sources[:top_k]
    actual_docs = [str(item.get("doc_id") or "") for item in selected]
    actual_chunks = [str(item.get("chunk_id") or "") for item in selected]
    relevant = set(expected_chunk_ids or expected_doc_ids)
    ranked = actual_chunks if expected_chunk_ids else actual_docs
    hits = [index for index, value in enumerate(ranked, start=1) if value in relevant]
    recall = len(set(ranked).intersection(relevant)) / len(relevant) if relevant else None
    precision = len([value for value in ranked if value in relevant]) / len(ranked) if ranked and relevant else None
    mrr = 1.0 / hits[0] if hits else (0.0 if relevant else None)
    dcg = sum(1.0 / log2(index + 1) for index in hits)
    ideal = sum(1.0 / log2(index + 1) for index in range(1, min(len(relevant), len(ranked)) + 1))
    ndcg = dcg / ideal if ideal else None
    duplicate_rate = (len(ranked) - len(set(ranked))) / max(1, len(ranked))
    scores = {
        "retrieval_hit_at_k": 1.0 if hits else (0.0 if relevant else None),
        "retrieval_recall_at_k": _round(recall),
        "retrieval_precision_at_k": _round(precision),
        "retrieval_mrr": _round(mrr),
        "retrieval_ndcg": _round(ndcg),
        "retrieval_duplicate_rate": _round(duplicate_rate),
    }
    all_ranked = [str(item.get("chunk_id") or "") for item in sources] if expected_chunk_ids else [
        str(item.get("doc_id") or "") for item in sources
    ]
    for fixed_k in sorted({max(1, int(value)) for value in fixed_ks}):
        fixed_ranked = all_ranked[:fixed_k]
        fixed_hits = [index for index, value in enumerate(fixed_ranked, start=1) if value in relevant]
        fixed_recall = len(set(fixed_ranked).intersection(relevant)) / len(relevant) if relevant else None
        scores[f"retrieval_hit_at_{fixed_k}"] = 1.0 if fixed_hits else (0.0 if relevant else None)
        scores[f"retrieval_recall_at_{fixed_k}"] = _round(fixed_recall)
    mrr_ranked = all_ranked[: max(1, int(mrr_k))]
    mrr_hits = [index for index, value in enumerate(mrr_ranked, start=1) if value in relevant]
    scores[f"retrieval_mrr_at_{max(1, int(mrr_k))}"] = _round(
        1.0 / mrr_hits[0] if mrr_hits else (0.0 if relevant else None)
    )
    failures = ["expected_evidence_not_retrieved"] if relevant and not hits else []
    records = records_for_scores(scores, "retrieval-v1", failures)
    return scores, failures, records


def evaluate_answer(
    *,
    answer: str,
    refused: bool,
    expected_refused: bool | None,
    expected_keywords: list[str],
    sources: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    keyword_score = None if not expected_keywords else float(all(keyword in answer for keyword in expected_keywords))
    refusal_score = None if expected_refused is None else float(bool(refused) == bool(expected_refused))
    citation_coverage = None
    if sources and answer.strip():
        citation_coverage = float(any(str(item.get("evidence_text") or item.get("snippet") or "").strip() for item in sources))
    scores = {
        "answer_keyword_coverage": keyword_score,
        "refusal_accuracy": refusal_score,
        "citation_context_available": citation_coverage,
    }
    failures = []
    if keyword_score == 0:
        failures.append("answer_keywords_missing")
    if refusal_score == 0:
        failures.append("refusal_mismatch")
    cited_numbers = [int(value) for value in re.findall(r"(?:证据|Evidence)\s*(\d+)", answer, flags=re.I)]
    if any(number < 1 or number > len(sources) for number in cited_numbers):
        failures.append("nonexistent_citation:answer")
    return scores, failures, records_for_scores(scores, "answer-rules-v1", failures)


def evaluate_outcome(
    actual: dict[str, Any], expected: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    if not expected:
        return {"outcome_match": None}, [], records_for_scores({"outcome_match": None}, "outcome-v1", [])
    mismatches = []
    for key, value in expected.items():
        if key == "appointment_created_after_approval":
            created = any(
                call.get("tool_name") == "appointment_book" and call.get("ok")
                for call in (actual.get("trajectory") or {}).get("calls") or []
            )
            if bool(value) != bool(created):
                mismatches.append(key)
            continue
        if key == "status":
            status = actual.get("status")
            if value == "pending_confirmation":
                status = "pending_confirmation" if actual.get("human_confirmation") else status
            if str(status or "") != str(value):
                mismatches.append(key)
            continue
        if actual.get(key) != value:
            mismatches.append(key)
    score = 0.0 if mismatches else 1.0
    failures = ["outcome_mismatch:" + ",".join(mismatches)] if mismatches else []
    scores = {"outcome_match": score}
    return scores, failures, records_for_scores(scores, "outcome-v1", failures)


def performance_scores(trajectory: dict[str, Any]) -> dict[str, Any]:
    calls = trajectory.get("calls") or []
    return {
        "input_tokens": int(trajectory.get("input_tokens") or 0),
        "output_tokens": int(trajectory.get("output_tokens") or 0),
        "total_tokens": int(trajectory.get("total_tokens") or 0),
        "latency_ms": int(trajectory.get("latency_ms") or 0),
        "tool_call_count": len(calls),
    }


def records_for_scores(scores: dict[str, Any], version: str, failures: list[str]) -> list[dict[str, Any]]:
    records = []
    for name, value in scores.items():
        record = score_record(name, None if value is None else float(value), failures)
        records.append(
            {
                **record,
                "evaluator_version": version,
            }
        )
    return records


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)
