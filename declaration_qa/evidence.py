from __future__ import annotations

from typing import Any
from datetime import date

from declaration_qa.domain import EvidenceDecision


DEFAULT_TOP_SCORE_THRESHOLD = 0.18
DEFAULT_RERANK_THRESHOLD = 0.12


def decide_evidence(sources: list[dict[str, Any]], filters: dict[str, Any], *, intent: str) -> EvidenceDecision:
    if not sources:
        return EvidenceDecision(ok=False, reason="no_matched_source", top_score=0.0)

    top = sources[0]
    top_score = float(top.get("score") or 0.0)
    rerank_score = float(top.get("rerank_score") or 0.0)
    if top_score < DEFAULT_TOP_SCORE_THRESHOLD and rerank_score < DEFAULT_RERANK_THRESHOLD:
        return EvidenceDecision(ok=False, reason="low_relevance", top_score=top_score)

    for key in ["biz_type", "region_code", "policy_version"]:
        expected = str(filters.get(key) or "").strip()
        if expected and str(top.get(key) or "").strip() != expected:
            return EvidenceDecision(ok=False, reason=f"{key}_mismatch", top_score=top_score)

    as_of = parse_date(filters.get("as_of")) or date.today()
    if not source_effective(top, as_of):
        return EvidenceDecision(ok=False, reason="policy_not_effective", top_score=top_score)

    if intent == "reject":
        for key in ["material_code", "reject_code"]:
            expected = str(filters.get(key) or "").strip()
            if expected and str(top.get(key) or "").strip() != expected:
                return EvidenceDecision(ok=False, reason=f"{key}_mismatch", top_score=top_score)
        conflict = conflict_reason(sources, keys=["material_code", "reject_code", "audit_node"])
        if conflict:
            return EvidenceDecision(ok=False, reason=conflict, top_score=top_score)
    else:
        conflict = conflict_reason(sources, keys=["policy_version"])
        if conflict:
            return EvidenceDecision(ok=False, reason=conflict, top_score=top_score)

    return EvidenceDecision(ok=True, reason="", top_score=top_score)


def conflict_reason(sources: list[dict[str, Any]], *, keys: list[str]) -> str:
    high_confidence = [source for source in sources if float(source.get("score") or 0.0) >= DEFAULT_TOP_SCORE_THRESHOLD]
    for key in keys:
        values = {str(source.get(key) or "").strip() for source in high_confidence if str(source.get(key) or "").strip()}
        if len(values) > 1:
            return f"conflicting_{key}"
    return ""


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
