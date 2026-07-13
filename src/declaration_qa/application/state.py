"""申报问答 LangGraph 状态模型。"""

from __future__ import annotations

from typing import Any, TypedDict


class DeclarationQAState(TypedDict, total=False):
    """申报问答工作流中的可变状态。"""
    user_id: int
    session_id: str
    query: str
    normalized_query: str
    query_image: dict[str, Any] | None
    query_image_ocr_text: str
    query_image_ocr_status: dict[str, Any]
    session_messages: list[dict[str, Any]]
    short_term_context: dict[str, Any]
    short_memory: dict[str, Any]
    long_term_context: dict[str, Any]

    agent_iteration: int
    agent_next: str
    agent_plan: dict[str, Any]
    agent_query_plan: dict[str, Any]
    agent_pending_tool_calls: list[dict[str, Any]]
    agent_call_signatures: list[str]
    agent_messages: list[dict[str, Any]]
    answer_mode: str
    agent_final_answer: str
    agent_final_prompt: str
    agent_answer_provider: str
    agent_final_model: str
    agent_final_token_usage: dict[str, int]
    agent_final_latency_ms: int | None
    frozen_knowledge_query: str
    agent_selected_item_code: str
    agent_selected_application_id: int | None
    tool_results: list[dict[str, Any]]

    region_code: str
    app_version: str
    policy_version: str
    biz_type: str | None
    request_biz_type: str | None
    metadata_filter: dict[str, Any]

    intent: str
    sub_intent: str
    route: str
    route_decision: dict[str, Any]
    retrieval_plan: dict[str, Any]
    retrieval_chain: str
    retrieval_query: str
    rag_query: str
    retrieval_queries: list[dict[str, Any]]

    business_snapshot: dict[str, Any] | None

    retrieval_filter: dict[str, Any]
    retrieved_sources: list[dict[str, Any]]
    rag_status: dict[str, Any]
    refuse_reason: str | None
    answer_generation_plan: dict[str, Any]

    answer: str
    refused: bool
    suggested_actions: list[dict[str, Any]]
    events: list[dict[str, Any]]
    audit_payload: dict[str, Any]
    audit_id: str
    langfuse_trace_id: str
