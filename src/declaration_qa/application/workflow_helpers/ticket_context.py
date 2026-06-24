"""办件上下文、政策过滤和业务快照辅助函数。"""

from __future__ import annotations

from typing import Any

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.workflow_helpers.collections import unique_strings
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.retrieval import normalize_retrieval_filter


def ticket_summary_from_context(context: dict[str, Any]) -> dict[str, Any]:
    """从办件上下文生成摘要。"""
    return {
        "ticket_id": str(context.get("apply_id") or ""),
        "biz_type": str(context.get("biz_type") or ""),
        "region_code": str(context.get("region_code") or ""),
        "status": str(context.get("status") or ""),
        "current_node": str(context.get("current_node") or ""),
        "node_name": str(context.get("node_name") or ""),
        "title": business_title(context),
        "updated_at": str(context.get("updated_at") or ""),
    }


def ticket_title_from_biz_type(biz_type: str) -> str:
    """根据业务类型生成办件名称。"""
    names = {
        "health_certificate": "健康证办理",
        "residence_permit": "居住证办理",
        "id_card_replace": "身份证换领",
    }
    return names.get(str(biz_type or ""), str(biz_type or "政务服务事项"))


def ticket_rag_query(state: DeclarationQAState) -> str:
    """构造办件增强检索查询文本。"""
    business = state.get("business_status") or state.get("reject_info") or {}
    apply_context = state.get("apply_context") or {}
    mapping = state.get("material_mapping") or {}
    parts = [
        business_title(business, apply_context, state),
        business.get("node_name"),
        business.get("current_node"),
        business.get("reject_reason"),
        business.get("material_name"),
        business.get("reject_code"),
        mapping.get("material_code"),
        state.get("normalized_query"),
    ]
    return normalize_text(" ".join(str(item) for item in parts if item))


def knowledge_rag_query(state: DeclarationQAState) -> str:
    """构造普通知识检索查询，已选办件时继承事项范围。"""
    query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
    if metadata_filter_has_item_scope(state.get("retrieval_filter") or {}):
        return query
    scope = selected_ticket_scope_text(state)
    if not scope:
        return query
    return normalize_text(f"{scope} {query}")


def selected_ticket_scope_text(state: DeclarationQAState) -> str:
    """从当前办件提取事项范围，不包含节点、退回原因等工单事实。"""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return ""
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    parts = [
        business_title(apply_context, state),
        payload.get("official_item_name"),
        payload.get("item_name"),
        payload.get("title"),
        payload.get("catalog_name"),
        payload.get("source_file"),
        payload.get("item_code"),
        apply_context.get("biz_type"),
    ]
    return normalize_text(" ".join(unique_strings([str(item) for item in parts if item])))


def metadata_filter_has_item_scope(filters: dict[str, Any]) -> bool:
    """判断检索条件是否已具备事项级约束。"""
    return any(filters.get(key) for key in ("item_code", "ssqd_code", "ssqd_id", "item_name"))


def business_title(*sources: dict[str, Any]) -> str:
    """生成业务类型展示名称。"""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("title", "item_name", "catalog_name"):
            title = str(source.get(key) or "").strip()
            if title:
                return title
        payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
        for key in ("item_name", "title", "catalog_name"):
            title = str(payload.get(key) or "").strip()
            if title:
                return title
    for source in sources:
        if isinstance(source, dict):
            title = ticket_title_from_biz_type(str(source.get("biz_type") or ""))
            if title:
                return title
    return "政务服务事项"


def base_policy_filter(state: DeclarationQAState, *, allow_request_filter: bool = True) -> dict[str, Any]:
    """构造基础政策检索过滤条件。"""
    filters: dict[str, Any] = {}
    if allow_request_filter:
        filters.update(normalize_retrieval_filter(state.get("metadata_filter") or {}))
    filters.update(ticket_metadata_filter(state))
    return normalize_retrieval_filter(filters)


def ticket_metadata_filter(state: DeclarationQAState) -> dict[str, Any]:
    """从当前办件上下文生成知识库可执行的事项过滤条件。"""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return {}
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    for key in ("item_code", "ssqd_code", "ssqd_id"):
        value = first_non_empty(apply_context.get(key), payload.get(key))
        if value:
            return {key: value}
    item_name = first_non_empty(
        apply_context.get("item_name"),
        payload.get("official_item_name"),
        payload.get("item_name"),
        apply_context.get("title"),
        payload.get("title"),
    )
    if item_name:
        return {"item_name": item_name}
    return {}


def first_non_empty(*values: Any) -> str:
    """返回第一个非空字符串值。"""
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_business_snapshot(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """构造可用于回答的办件快照。"""
    cleaned = {key: value for key, value in (payload or {}).items() if value not in ("", None, [], {})}
    if not cleaned:
        return {}
    return {
        "type": kind,
        "source": "business_adapter",
        "apply_id": cleaned.get("apply_id"),
        "biz_type": cleaned.get("biz_type"),
        "title": business_title(cleaned),
        "region_code": cleaned.get("region_code"),
        "status": cleaned.get("status"),
        "current_node": cleaned.get("current_node"),
        "node_name": cleaned.get("node_name"),
        "material_code": cleaned.get("material_code"),
        "material_name": cleaned.get("material_name"),
        "reject_code": cleaned.get("reject_code"),
        "reject_reason": cleaned.get("reject_reason"),
        "audit_node": cleaned.get("audit_node"),
        "submitted_at": cleaned.get("submitted_at"),
        "completed_at": cleaned.get("completed_at"),
        "updated_at": cleaned.get("updated_at"),
        "payload": redact_business_payload(cleaned.get("payload") or {}),
    }


def redact_business_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """脱敏业务载荷中的敏感字段。"""
    redacted = {}
    for key, value in (payload or {}).items():
        lowered = str(key).lower()
        if (
            any(token in lowered for token in ["id_card", "idcard", "phone", "mobile"])
            or lowered in {"name", "applicant_name", "contact_name", "person_name"}
            or lowered.endswith("_address")
        ):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted

