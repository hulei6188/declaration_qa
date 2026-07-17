"""构建知识证据与业务证据的稳定编号映射。"""

from __future__ import annotations

from typing import Any


BUSINESS_PRIMARY_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "ticket_list_query": ("item_code",),
    "ticket_detail_query": ("application_id",),
    "appointment_list_query": ("appointment_id",),
    "appointment_hall_query": ("hall_code",),
    "appointment_slot_query": ("slot_id",),
    "appointment_book": ("appointment_id",),
    "appointment_cancel": ("appointment_id",),
    "queue_status_query": ("appointment_id",),
    "payment_order_query": ("payment_order_id",),
    "payment_receipt_query": ("payment_receipt_id",),
    "submission_receipt_query": ("submission_receipt_id",),
    "result_certificate_query": ("result_certificate_id",),
    "delivery_status_query": ("delivery_id",),
    "consultation_record_query": ("consultation_id",),
}


def build_evidence_registry(
    sources: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """把展示编号绑定到可审计的底层证据标识。"""
    knowledge = [entry for entry, _source in registered_knowledge_sources(sources)]
    business = [entry for entry, _result in registered_business_results(tool_results)]
    return {"knowledge": knowledge, "business": business}


def registered_knowledge_sources(
    sources: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """返回具有块标识的知识来源及其稳定编号。"""
    registered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        chunk_id = str(source.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        entry = {
            "evidence_id": f"证据{len(registered) + 1}",
            "chunk_id": chunk_id,
        }
        if source.get("memory_origin"):
            entry["origin"] = str(source.get("memory_origin"))
        if source.get("source_turn_id"):
            entry["source_turn_id"] = str(source.get("source_turn_id"))
        registered.append((entry, source))
    return registered


def registered_business_results(
    results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """返回同时具备调用标识和业务主键的成功业务结果。"""
    registered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for result in results:
        if not isinstance(result, dict) or result.get("tool_name") == "knowledge_search":
            continue
        if not result.get("ok") or not isinstance(result.get("data"), dict):
            continue
        call_id = str(result.get("call_id") or "").strip()
        tool_name = str(result.get("tool_name") or "").strip()
        business_key = extract_business_key(tool_name, result["data"])
        if not call_id or not tool_name or not business_key:
            continue
        entry = {
            "evidence_id": f"业务{len(registered) + 1}",
            "call_id": call_id,
            "tool_name": tool_name,
            "business_key": business_key,
        }
        if result.get("memory_origin"):
            entry["origin"] = str(result.get("memory_origin"))
        if result.get("observed_at"):
            entry["observed_at"] = str(result.get("observed_at"))
        if result.get("source_turn_id"):
            entry["source_turn_id"] = str(result.get("source_turn_id"))
        registered.append((entry, result))
    return registered


def extract_business_key(tool_name: str, data: dict[str, Any]) -> dict[str, list[Any]]:
    """按工具契约提取返回实体的业务主键。"""
    fields = BUSINESS_PRIMARY_KEY_FIELDS.get(str(tool_name or ""), ())
    output: dict[str, list[Any]] = {}
    for field in fields:
        values = collect_field_values(data, field)
        if values:
            output[field] = values
    return output


def collect_field_values(value: Any, field: str) -> list[Any]:
    """递归收集指定字段，并按首次出现顺序去重。"""
    collected: list[Any] = []

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            if field in current and valid_business_key_value(current[field]):
                append_unique(collected, current[field])
            for nested in current.values():
                visit(nested)
        elif isinstance(current, list):
            for nested in current:
                visit(nested)

    visit(value)
    return collected


def valid_business_key_value(value: Any) -> bool:
    """只接受可序列化且非空的标量主键。"""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return value > 0


def append_unique(values: list[Any], value: Any) -> None:
    """保留业务主键原始类型并避免重复。"""
    if value not in values:
        values.append(value)
