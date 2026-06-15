from __future__ import annotations

from datetime import date
from typing import Any

from declaration_qa.domain import EvidenceDecision
from declaration_qa.retrieval import (
    KNOWLEDGE_RETRIEVAL_CHAIN,
    TICKET_AUGMENTED_CHAIN,
    metadata_coverage,
)


DEFAULT_TOP_SCORE_THRESHOLD = 0.18
DEFAULT_RERANK_THRESHOLD = 0.12
MIN_EVIDENCE_SOURCES = 1


def decide_evidence(
    sources: list[dict[str, Any]],
    filters: dict[str, Any],
    *,
    intent: str,
    chain: str = KNOWLEDGE_RETRIEVAL_CHAIN,
    business_payload: dict[str, Any] | None = None,
) -> EvidenceDecision:
    coverage = metadata_coverage(filters, chain=chain)
    checks: dict[str, Any] = {
        "chain": chain,
        "metadata_coverage": coverage,
        "business_payload_present": bool(business_payload),
    }
    if not sources:
        return EvidenceDecision(
            ok=False,
            reason="no_matched_source",
            top_score=0.0,
            source_count=0,
            required_metadata=coverage["required"],
            missing_metadata=coverage["missing"],
            checks=checks,
        )

    top = sources[0]
    top_score = float(top.get("score") or 0.0)
    rerank_score = float(top.get("rerank_score") or 0.0)
    checks.update({"top_score": top_score, "rerank_score": rerank_score})
    if top_score < DEFAULT_TOP_SCORE_THRESHOLD and rerank_score < DEFAULT_RERANK_THRESHOLD:
        return EvidenceDecision(
            ok=False,
            reason="low_relevance",
            top_score=top_score,
            source_count=len(sources),
            required_metadata=coverage["required"],
            missing_metadata=coverage["missing"],
            checks=checks,
        )

    if len(sources) < MIN_EVIDENCE_SOURCES:
        return EvidenceDecision(
            ok=False,
            reason="insufficient_evidence",
            top_score=top_score,
            source_count=len(sources),
            required_metadata=coverage["required"],
            missing_metadata=coverage["missing"],
            checks=checks,
        )

    if not source_traceable(top):
        return EvidenceDecision(
            ok=False,
            reason="source_not_traceable",
            top_score=top_score,
            source_count=len(sources),
            required_metadata=coverage["required"],
            missing_metadata=coverage["missing"],
            checks=checks,
        )

    if chain == TICKET_AUGMENTED_CHAIN and not business_payload:
        return EvidenceDecision(
            ok=False,
            reason="missing_business_snapshot",
            top_score=top_score,
            source_count=len(sources),
            required_metadata=coverage["required"],
            missing_metadata=coverage["missing"],
            checks=checks,
        )

    return EvidenceDecision(
        ok=True,
        reason="",
        top_score=top_score,
        source_count=len(sources),
        required_metadata=coverage["required"],
        missing_metadata=coverage["missing"],
        checks=checks,
    )


def conflict_reason(sources: list[dict[str, Any]], *, keys: list[str]) -> str:
    high_confidence = [source for source in sources if float(source.get("score") or 0.0) >= DEFAULT_TOP_SCORE_THRESHOLD]
    for key in keys:
        values = {str(source.get(key) or "").strip() for source in high_confidence if str(source.get(key) or "").strip()}
        if len(values) > 1:
            return f"conflicting_{key}"
    return ""


def source_traceable(source: dict[str, Any]) -> bool:
    if str(source.get("doc_id") or "").strip() and str(source.get("title") or "").strip():
        return True
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    return bool(
        str(source.get("source_url") or "").strip()
        or str(metadata.get("item_code") or "").strip()
        or str(metadata.get("item_name") or "").strip()
        or str(metadata.get("catalog_name") or "").strip()
        or str(metadata.get("catalog_subitem_name") or "").strip()
    )


def source_effective(source: dict[str, Any], as_of: date) -> bool:
    start = parse_date(source.get("effective_start_time"))
    end = parse_date(source.get("effective_end_time"))
    if start and start > as_of:
        return False
    if end and end <= as_of:
        return False
    return True


def parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None
