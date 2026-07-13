"""短期知识证据引用的可信回填与候选合并。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from declaration_qa.core.business import region_code_matches
from declaration_qa.core.domain import RetrievedSource
from declaration_qa.rag.engine.scoring import matches_filter
from declaration_qa.rag.retrieval import published_visibility_filter


def trusted_historical_evidence_candidates(
    rag_engine: Any,
    state: dict[str, Any],
    *,
    filters: dict[str, Any],
    session_id: str,
    region_code: str,
) -> tuple[list[RetrievedSource], dict[str, Any]]:
    """从当前会话短期记忆加载并校验历史知识证据引用。"""
    short_memory = state.get("short_memory") if isinstance(state.get("short_memory"), dict) else {}
    refs = [
        item
        for item in (short_memory.get("recent_knowledge_evidence_refs") or [])
        if isinstance(item, dict)
    ][:12]
    status: dict[str, Any] = {
        "requested_count": len(refs),
        "accepted_count": 0,
        "loaded_count": 0,
        "rejected_count": 0,
        "rejected_reasons": {},
    }
    accepted: list[dict[str, Any]] = []
    for ref in refs:
        reason = historical_ref_denial_reason(
            ref,
            session_id=session_id,
            region_code=region_code,
            filters=filters,
        )
        if reason:
            status["rejected_count"] += 1
            reasons = status["rejected_reasons"]
            reasons[reason] = int(reasons.get(reason) or 0) + 1
        else:
            accepted.append(ref)
    status["accepted_count"] = len(accepted)
    loader = getattr(rag_engine, "get_chunks_by_ids", None)
    if not callable(loader) or not accepted:
        return [], status
    chunks_by_id = {
        chunk.id: chunk
        for chunk in loader([str(item.get("chunk_id") or "") for item in accepted])
    }
    candidates: list[RetrievedSource] = []
    for ref in accepted:
        chunk = chunks_by_id.get(str(ref.get("chunk_id") or ""))
        denial = (
            historical_chunk_denial_reason(chunk, ref, region_code=region_code, filters=filters)
            if chunk is not None
            else "chunk_missing"
        )
        if denial:
            status["rejected_count"] += 1
            reasons = status["rejected_reasons"]
            reasons[denial] = int(reasons.get(denial) or 0) + 1
            continue
        candidates.append(
            RetrievedSource(
                chunk=chunk,
                score=0.01,
                retrieval_channel="trusted_history_ref",
            )
        )
    status["loaded_count"] = len(candidates)
    return candidates, status


def historical_ref_denial_reason(
    ref: dict[str, Any],
    *,
    session_id: str,
    region_code: str,
    filters: dict[str, Any],
) -> str:
    """校验引用属于当前会话且仍在有效期内。"""
    if not str(ref.get("evidence_ref") or "").startswith("ev_"):
        return "invalid_ref"
    if str(ref.get("session_id") or "") != str(session_id or ""):
        return "session_mismatch"
    try:
        expires_at = datetime.fromisoformat(str(ref.get("expires_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return "invalid_expiry"
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        return "expired"
    expected_region = str(filters.get("region_code") or region_code or "")
    ref_region = str(ref.get("region_code") or "")
    if expected_region and ref_region and not region_code_matches(ref_region, expected_region):
        return "region_mismatch"
    expected_version = str(filters.get("policy_version") or "")
    ref_version = str(ref.get("policy_version") or "")
    if expected_version and ref_version and expected_version != ref_version:
        return "policy_version_mismatch"
    return ""


def historical_chunk_denial_reason(
    chunk: Any,
    ref: dict[str, Any],
    *,
    region_code: str,
    filters: dict[str, Any],
) -> str:
    """回表后重新校验过滤条件、文档、地区、版本与生效日期。"""
    if not matches_filter(chunk, published_visibility_filter(filters)):
        return "filter_mismatch"
    if str(chunk.doc_id or "") != str(ref.get("document_id") or ""):
        return "document_mismatch"
    expected_region = str(filters.get("region_code") or region_code or "")
    ref_region = str(ref.get("region_code") or "")
    if ref_region and chunk.region_code and not region_code_matches(chunk.region_code, ref_region):
        return "region_mismatch"
    if expected_region and chunk.region_code and not region_code_matches(chunk.region_code, expected_region):
        return "region_mismatch"
    expected_version = str(filters.get("policy_version") or "")
    ref_version = str(ref.get("policy_version") or "")
    if ref_version and chunk.policy_version and str(chunk.policy_version) != ref_version:
        return "policy_version_mismatch"
    if expected_version and chunk.policy_version and str(chunk.policy_version) != expected_version:
        return "policy_version_mismatch"
    effective_date = parse_as_of_date(filters.get("as_of")) or date.today()
    if chunk.effective_start_time and chunk.effective_start_time > effective_date:
        return "not_yet_effective"
    if chunk.effective_end_time and chunk.effective_end_time <= effective_date:
        return "expired_policy"
    return ""


def parse_as_of_date(value: Any) -> date | None:
    """解析检索的历史生效日期。"""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value or ""))
    except ValueError:
        return None


def merge_retrieved_source_candidates(
    current: list[RetrievedSource],
    historical: list[RetrievedSource],
) -> list[RetrievedSource]:
    """合并当前召回与历史回填，同一 chunk 优先当前召回分数。"""
    merged: dict[str, RetrievedSource] = {}
    for source in [*historical, *current]:
        existing = merged.get(source.chunk.id)
        if existing is None or float(source.score or 0.0) >= float(existing.score or 0.0):
            merged[source.chunk.id] = source
    return list(merged.values())
