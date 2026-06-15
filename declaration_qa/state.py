from __future__ import annotations

from typing import Any, TypedDict


class DeclarationQAState(TypedDict, total=False):
    user_id: int
    session_id: str
    query: str
    normalized_query: str

    region_code: str
    app_version: str
    policy_version: str
    biz_type: str | None
    apply_id: str | None
    apply_context: dict[str, Any] | None

    intent: str
    route: str

    business_status: dict[str, Any] | None
    reject_info: dict[str, Any] | None
    material_mapping: dict[str, Any] | None

    retrieval_filter: dict[str, Any]
    retrieved_sources: list[dict[str, Any]]
    rag_status: dict[str, Any]
    evidence_ok: bool
    refuse_reason: str | None

    answer: str
    refused: bool
    suggested_actions: list[dict[str, Any]]
    events: list[dict[str, Any]]
    audit_payload: dict[str, Any]
    audit_id: str
