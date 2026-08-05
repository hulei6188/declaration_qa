"""按一级领域动态裁剪 Agent 工具集。"""

from __future__ import annotations

from typing import Any


DOMAIN_TOOL_POLICY: dict[str, frozenset[str]] = {
    "knowledge": frozenset({"knowledge_search"}),
    "business_query": frozenset(
        {
            "knowledge_search",
            "ticket_list_query",
            "ticket_detail_query",
            "appointment_list_query",
            "queue_status_query",
            "payment_order_query",
            "payment_receipt_query",
            "submission_receipt_query",
            "result_certificate_query",
            "delivery_status_query",
            "consultation_record_query",
        }
    ),
    "application": frozenset(
        {
            "knowledge_search",
            "application_submit",
        }
    ),
    "appointment": frozenset(
        {
            "knowledge_search",
            "appointment_list_query",
            "appointment_hall_query",
            "appointment_slot_query",
            "appointment_book",
            "appointment_cancel",
        }
    ),
    "complaint": frozenset({"knowledge_search", "consultation_record_query", "complaint_create"}),
    "conversation": frozenset(),
    "clarification": frozenset(),
}


WRITE_TOOLS = frozenset({"application_submit", "appointment_book", "appointment_cancel", "complaint_create"})

OPERATION_TOOL_POLICY: dict[tuple[str, str], frozenset[str]] = {
    ("application", "start"): frozenset({"knowledge_search", "application_submit"}),
    ("application", "continue"): frozenset({"knowledge_search", "application_submit"}),
    ("appointment", "start"): frozenset(
        {"knowledge_search", "appointment_hall_query", "appointment_slot_query", "appointment_book"}
    ),
    ("appointment", "book"): frozenset(
        {"knowledge_search", "appointment_hall_query", "appointment_slot_query", "appointment_book"}
    ),
    ("appointment", "cancel"): frozenset({"appointment_list_query", "appointment_cancel"}),
    ("complaint", "start"): frozenset({"knowledge_search", "complaint_create"}),
    ("complaint", "continue"): frozenset({"knowledge_search", "complaint_create"}),
    ("complaint", "create"): frozenset({"knowledge_search", "complaint_create"}),
}


def allowed_tool_names(route_decision: dict[str, Any] | None) -> frozenset[str]:
    """返回当前结构化路由唯一允许的工具名称。"""
    decision = route_decision if isinstance(route_decision, dict) else {}
    if decision.get("needs_clarification"):
        return frozenset()
    domain = str(decision.get("domain") or "")
    operation = str(decision.get("operation") or "")
    operation_scope = OPERATION_TOOL_POLICY.get((domain, operation))
    if operation_scope is not None:
        return operation_scope
    allowed = DOMAIN_TOOL_POLICY.get(domain, frozenset())
    if domain in {"application", "appointment", "complaint"}:
        allowed = allowed - WRITE_TOOLS
    return frozenset(allowed)


def filter_tool_definitions_for_route(
    definitions: list[dict[str, Any]],
    route_decision: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """只向模型暴露当前领域白名单中的工具。"""
    allowed = allowed_tool_names(route_decision)
    return [item for item in definitions if tool_definition_name(item) in allowed]


def partition_tool_calls(
    calls: list[dict[str, Any]],
    route_decision: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """在执行前再次划分允许和越权调用。"""
    allowed = allowed_tool_names(route_decision)
    accepted, denied = [], []
    for call in calls:
        (accepted if str(call.get("name") or "") in allowed else denied).append(call)
    return accepted, denied


def tool_definition_name(item: dict[str, Any]) -> str:
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    return str(item.get("name") or function.get("name") or "")
