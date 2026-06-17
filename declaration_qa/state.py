"""申报问答 LangGraph 状态模型。"""

from __future__ import annotations

from typing import Any, TypedDict


class DeclarationQAState(TypedDict, total=False):
    """申报问答工作流中的可变状态。"""
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
    metadata_filter: dict[str, Any]

    intent: str
    sub_intent: str
    route: str
    route_decision: dict[str, Any]
    question_plan: dict[str, Any]
    requires_ticket_context: bool
    multi_question: bool
    sub_tasks: list[dict[str, Any]]
    retrieval_chain: str
    need_ticket_selection: bool
    ticket_selection_reason: str
    tickets: list[dict[str, Any]]
    ticket_context: dict[str, Any] | None
    rag_query: str

    business_status: dict[str, Any] | None
    reject_info: dict[str, Any] | None
    business_snapshot: dict[str, Any] | None
    material_mapping: dict[str, Any] | None

    retrieval_filter: dict[str, Any]
    retrieved_sources: list[dict[str, Any]]
    rag_status: dict[str, Any]
    evidence_ok: bool
    evidence_decision: dict[str, Any]
    refuse_reason: str | None

    answer: str
    refused: bool
    suggested_actions: list[dict[str, Any]]
    events: list[dict[str, Any]]
    audit_payload: dict[str, Any]
    audit_id: str
