"""Milvus 元数据行和过滤表达式。"""

from __future__ import annotations

from datetime import date, datetime
import json
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
MILVUS_METADATA_MAX_BYTES = 48 * 1024

# 完整表格结构由 PostgreSQL chunk store 保存；Milvus 只需要支持过滤和命中定位。
MILVUS_METADATA_EXCLUDED_FIELDS = frozenset(
    {
        "docling_chunk_specs",
        "llm_table_context",
        "table_html",
        "table_json",
        "table_text",
        "asset_text",
        "ocr_text",
        "visual_summary",
    }
)
MILVUS_METADATA_FALLBACK_FIELDS = (
    "chunk_type",
    "semantic_type",
    "section_path",
    "section_title",
    "node_title",
    "parent_section_id",
    "node_order",
    "attachment_name",
    "attachment_type",
    "attachment_page_no",
    "attachment_page_numbers",
    "attachment_page_start",
    "attachment_page_end",
    "asset_id",
    "asset_kind",
    "asset_url",
    "thumbnail_url",
    "source_attachment_chunk_type",
    "table_index",
    "table_source",
    "table_row_count",
    "table_column_count",
    "table_headers",
    "form_field_name",
    "form_field_label",
    "form_field_type",
    "form_field_options",
    "form_field_required",
    "document_parser",
    "extraction_method",
    "modality",
    "visual_summary_status",
)


def milvus_metadata_row(chunk: KnowledgeChunk) -> dict[str, Any]:
    """将知识片段转换为 Milvus 存储行。"""
    raw_metadata = dict(chunk.metadata or {})
    metadata = compact_milvus_metadata(raw_metadata)
    values = {
        "region_code": chunk.region_code or raw_metadata.get("region_code") or "",
        "region_key": raw_metadata.get("region_key") or chunk.region_code or "",
        "region_name": raw_metadata.get("region_name") or "",
        "policy_version": chunk.policy_version or raw_metadata.get("policy_version") or "",
        "source_version": raw_metadata.get("source_version") or chunk.policy_version or "",
        "official_source_url": raw_metadata.get("official_source_url") or chunk.source_url or "",
        "catalog_name": raw_metadata.get("catalog_name") or "",
        "catalog_subitem_name": raw_metadata.get("catalog_subitem_name") or "",
        "item_code": raw_metadata.get("item_code") or "",
        "item_name": raw_metadata.get("item_name") or "",
        "item_type": raw_metadata.get("item_type") or "",
        "theme_code": raw_metadata.get("theme_code") or "",
        "service_object": raw_metadata.get("service_object") or "",
        "basic_code": raw_metadata.get("basic_code") or "",
        "agency_name": raw_metadata.get("agency_name") or "",
        "ssqd_code": raw_metadata.get("ssqd_code") or "",
        "ssqd_id": raw_metadata.get("ssqd_id") or "",
        "audit_node": chunk.audit_node or raw_metadata.get("audit_node") or "",
        "status": chunk.status or raw_metadata.get("status") or "published",
        "doc_status": raw_metadata.get("doc_status") or chunk.status or raw_metadata.get("status") or "published",
        "chunking_strategy": raw_metadata.get("chunking_strategy") or "",
        "modality": raw_metadata.get("modality") or "text",
        "effective_start_time": raw_metadata.get("effective_start_time") or "",
        "effective_end_time": raw_metadata.get("effective_end_time") or "",
    }
    return {
        **{
            field_name: truncate_milvus_text(values.get(field_name), max_length)
            for field_name, max_length in MILVUS_TEXT_METADATA_FIELDS.items()
        },
        MILVUS_JSON_METADATA_FIELD: milvus_json_value(metadata),
    }


def compact_milvus_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """构建可安全写入 Milvus JSON 字段的元数据投影。"""
    compact = {
        str(key): milvus_json_value(value)
        for key, value in metadata.items()
        if str(key) not in MILVUS_METADATA_EXCLUDED_FIELDS
    }
    if milvus_metadata_size_bytes(compact) <= MILVUS_METADATA_MAX_BYTES:
        return compact

    fallback = {
        key: compact[key]
        for key in MILVUS_METADATA_FALLBACK_FIELDS
        if key in compact
    }
    fallback["milvus_metadata_truncated"] = True
    return compact_milvus_metadata_fallback(fallback)


def compact_milvus_metadata_fallback(metadata: dict[str, Any]) -> dict[str, Any]:
    """将异常大的元数据收敛为定位所需的轻量字段。"""
    compact = {
        key: compact_milvus_metadata_value(value, max_string_length=512, max_items=16)
        for key, value in metadata.items()
    }
    if milvus_metadata_size_bytes(compact) <= MILVUS_METADATA_MAX_BYTES:
        return compact
    return {"milvus_metadata_truncated": True}


def compact_milvus_metadata_value(value: Any, *, max_string_length: int, max_items: int) -> Any:
    """限制回退元数据的嵌套集合和文本长度。"""
    value = milvus_json_value(value)
    if isinstance(value, str):
        return value[:max_string_length]
    if isinstance(value, list):
        return [
            compact_milvus_metadata_value(item, max_string_length=max_string_length, max_items=max_items)
            for item in value[:max_items]
        ]
    if isinstance(value, dict):
        return {
            key: compact_milvus_metadata_value(item, max_string_length=max_string_length, max_items=max_items)
            for key, item in list(value.items())[:max_items]
        }
    return value


def milvus_metadata_size_bytes(value: dict[str, Any]) -> int:
    """计算 Milvus JSON 字段的 UTF-8 编码大小。"""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


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
        if key not in MILVUS_TEXT_METADATA_FIELDS and key != "doc_id":
            continue
        if value in ("", None, []):
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
