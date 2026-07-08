"""Milvus 元数据行和过滤表达式。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from declaration_qa.core.domain import KnowledgeChunk


MILVUS_TEXT_METADATA_FIELDS: dict[str, int] = {
    "region_code": 32,
    "region_key": 32,
    "region_name": 120,
    "policy_version": 80,
    "source_version": 120,
    "official_source_url": 1024,
    "catalog_name": 240,
    "catalog_subitem_name": 240,
    "item_code": 120,
    "item_name": 240,
    "item_type": 80,
    "theme_code": 80,
    "service_object": 120,
    "basic_code": 120,
    "agency_name": 240,
    "ssqd_code": 120,
    "ssqd_id": 120,
    "audit_node": 120,
    "status": 40,
    "doc_status": 40,
    "chunking_strategy": 40,
    "modality": 40,
    "effective_start_time": 32,
    "effective_end_time": 32,
}
MILVUS_JSON_METADATA_FIELD = "metadata"


def milvus_metadata_row(chunk: KnowledgeChunk) -> dict[str, Any]:
    """将知识片段转换为 Milvus 存储行。"""
    metadata = dict(chunk.metadata or {})
    values = {
        "region_code": chunk.region_code or metadata.get("region_code") or "",
        "region_key": metadata.get("region_key") or chunk.region_code or "",
        "region_name": metadata.get("region_name") or "",
        "policy_version": chunk.policy_version or metadata.get("policy_version") or "",
        "source_version": metadata.get("source_version") or chunk.policy_version or "",
        "official_source_url": metadata.get("official_source_url") or chunk.source_url or "",
        "catalog_name": metadata.get("catalog_name") or "",
        "catalog_subitem_name": metadata.get("catalog_subitem_name") or "",
        "item_code": metadata.get("item_code") or "",
        "item_name": metadata.get("item_name") or "",
        "item_type": metadata.get("item_type") or "",
        "theme_code": metadata.get("theme_code") or "",
        "service_object": metadata.get("service_object") or "",
        "basic_code": metadata.get("basic_code") or "",
        "agency_name": metadata.get("agency_name") or "",
        "ssqd_code": metadata.get("ssqd_code") or "",
        "ssqd_id": metadata.get("ssqd_id") or "",
        "audit_node": chunk.audit_node or metadata.get("audit_node") or "",
        "status": chunk.status or metadata.get("status") or "published",
        "doc_status": metadata.get("doc_status") or chunk.status or metadata.get("status") or "published",
        "chunking_strategy": metadata.get("chunking_strategy") or "",
        "modality": metadata.get("modality") or "text",
        "effective_start_time": metadata.get("effective_start_time") or "",
        "effective_end_time": metadata.get("effective_end_time") or "",
    }
    return {
        **{
            field_name: truncate_milvus_text(values.get(field_name), max_length)
            for field_name, max_length in MILVUS_TEXT_METADATA_FIELDS.items()
        },
        MILVUS_JSON_METADATA_FIELD: milvus_json_value(metadata),
    }


def truncate_milvus_text(value: Any, max_length: int) -> str:
    """按长度截断 Milvus 文本字段。"""
    text = "" if value is None else str(value)
    return text[:max_length]


def milvus_json_value(value: Any) -> Any:
    """将值序列化为 Milvus JSON 字段文本。"""
    if isinstance(value, dict):
        return {str(key): milvus_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [milvus_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [milvus_json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def milvus_filter_expr(filters: dict[str, Any]) -> str:
    """将检索过滤条件转换为 Milvus 表达式。"""
    expressions = []
    for key, value in (filters or {}).items():
        if key not in MILVUS_TEXT_METADATA_FIELDS or value in ("", None, []):
            continue
        if isinstance(value, list):
            values = [milvus_escape_string(item) for item in value if str(item or "").strip()]
            if not values:
                continue
            expressions.append(f"{key} in [{', '.join(values)}]")
        else:
            expressions.append(f"{key} == {milvus_escape_string(value)}")
    return " and ".join(expressions)


def milvus_escape_string(value: Any) -> str:
    """转义 Milvus 过滤表达式字符串。"""
    text = str(value or "").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'
