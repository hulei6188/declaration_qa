"""表格语义化 chunk 构建工具。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from declaration_qa.indexing.document_parsing.text import (
    clean_inline_text,
    compact_table_context,
    table_row_text,
)


MAX_TABLE_SEMANTIC_ROW_CHUNKS = 120
MAX_TABLE_SUMMARY_FIELDS = 40
MAX_TABLE_SUMMARY_VALUES = 12
MAX_TABLE_ROW_VALUE_CHARS = 900
MAX_TABLE_TITLE_CHARS = 500
MAX_TABLE_FIELD_NAME_CHARS = 300
MAX_TABLE_SOURCE_LINE_CHARS = 500
MAX_INDEXABLE_TABLE_CHUNK_CHARS = 6000


def build_table_semantic_chunk_specs(
    *,
    id_prefix: str,
    table_index: int,
    table_json_value: Any,
    table_text_value: str,
    table_html_value: str,
    base_metadata: dict[str, Any],
    source_lines: list[str],
    source_url: str = "",
    table_source: str = "",
    max_row_chunks: int = MAX_TABLE_SEMANTIC_ROW_CHUNKS,
) -> list[dict[str, Any]]:
    """为一个表格构建父块、摘要块和行/键值块。"""
    table_index = coerce_table_index(table_index, 0)
    table = normalized_table_payload(table_json_value)
    if not table:
        return []
    context = compact_table_context(table)
    parent_suffix = f"{id_prefix}:parent"
    parent_metadata = table_common_metadata(
        base_metadata=base_metadata,
        semantic_type="table_parent",
        table_index=table_index,
        table=table,
        table_text_value=table_text_value,
        table_html_value=table_html_value,
        context=context,
        table_source=table_source,
        indexable=False,
    )
    specs = [
        {
            "id_suffix": parent_suffix,
            "chunk_type": "parent",
            "content": table_parent_content(
                source_lines=source_lines,
                table=table,
                table_text_value=table_text_value,
            ),
            "source_url": source_url,
            "metadata": {
                **parent_metadata,
                "context_only": True,
                "context_role": "table_context",
            },
        }
    ]
    summary_content = table_summary_content(
        source_lines=source_lines,
        table=table,
        table_text_value=table_text_value,
    )
    summary_content = bounded_chunk_text(summary_content)
    specs.append(
        {
            "id_suffix": f"{id_prefix}:summary",
            "parent_id_suffix": parent_suffix,
            "chunk_type": "child",
            "content": summary_content,
            "source_url": source_url,
            "metadata": table_common_metadata(
                base_metadata=base_metadata,
                semantic_type="table_summary",
                table_index=table_index,
                table=table,
                table_text_value=table_text_value,
                table_html_value=table_html_value,
                context=context,
                table_source=table_source,
                indexable=True,
            ),
        }
    )
    row_items = semantic_table_rows(table)
    visible_rows = row_items[: max(0, int(max_row_chunks or 0))]
    for row_item in visible_rows:
        semantic_type = "table_kv" if row_item["is_key_value"] else "table_row"
        specs.append(
            {
                "id_suffix": f"{id_prefix}:{semantic_type}:{row_item['row_index']}",
                "parent_id_suffix": parent_suffix,
                "chunk_type": "child",
                "content": bounded_chunk_text(
                    table_row_chunk_content(
                        source_lines=source_lines,
                        table=table,
                        row_item=row_item,
                    )
                ),
                "source_url": source_url,
                "metadata": {
                    **table_common_metadata(
                        base_metadata=base_metadata,
                        semantic_type=semantic_type,
                        table_index=table_index,
                        table=table,
                        table_text_value="",
                        table_html_value="",
                        context=context,
                        table_source=table_source,
                        indexable=True,
                        include_table_json=False,
                    ),
                    "table_row_index": row_item["row_index"],
                    "table_row_values": row_item["values"],
                    "table_row_text": row_item["text"],
                    "source_table_parent_id_suffix": parent_suffix,
                    "table_row_chunks_truncated": len(row_items) > len(visible_rows),
                    **key_value_metadata(row_item),
                },
            }
        )
    return specs


def table_common_metadata(
    *,
    base_metadata: dict[str, Any],
    semantic_type: str,
    table_index: int,
    table: dict[str, Any],
    table_text_value: str,
    table_html_value: str,
    context: dict[str, Any],
    table_source: str,
    indexable: bool,
    include_table_json: bool = True,
) -> dict[str, Any]:
    """构建表格派生 chunk 的共享元数据。"""
    light_base_metadata = {
        key: value
        for key, value in base_metadata.items()
        if key not in {"table_json", "table_html", "table_text"}
    }
    metadata = {
        **light_base_metadata,
        "chunk_type": semantic_type,
        "table_index": table_index,
        "table_source": table_source,
        "table_row_count": table_row_count(table),
        "table_column_count": table_column_count(table),
        "table_headers": table_headers(table),
        "llm_table_context": context,
        "table_semantic_index": True,
        "indexable": indexable,
    }
    if table_text_value:
        metadata["table_text"] = table_text_value
    if table_html_value:
        metadata["table_html"] = table_html_value
    if include_table_json and table:
        metadata["table_json"] = table
    return metadata


def table_parent_content(*, source_lines: list[str], table: dict[str, Any], table_text_value: str) -> str:
    """生成整表父块内容。"""
    lines = [
        "结构化表格上下文",
        *non_empty_lines(source_lines),
        table_identity_line(table),
        table_size_line(table),
        "",
        "表格文本：",
        clean_inline_text(table_text_value) if table_text_value else table_summary_text(table),
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def table_summary_content(*, source_lines: list[str], table: dict[str, Any], table_text_value: str) -> str:
    """生成适合语义召回的表格摘要。"""
    key_values = [item for item in semantic_table_rows(table) if item["is_key_value"]]
    fields = [str(item.get("field_name") or "").strip() for item in key_values if str(item.get("field_name") or "").strip()]
    value_examples = [
        f"{item['field_name']}：{item['field_value']}"
        for item in key_values
        if item.get("field_name") and item.get("field_value")
    ]
    lines = [
        "结构化表格摘要",
        *non_empty_lines(source_lines),
        table_identity_line(table),
        table_size_line(table),
    ]
    headers = table_headers(table)
    if headers:
        lines.append(f"表头：{'、'.join(headers)}")
    if fields:
        lines.append(f"主要字段：{'、'.join(fields[:MAX_TABLE_SUMMARY_FIELDS])}")
    if value_examples:
        lines.append(f"关键内容：{'；'.join(value_examples[:MAX_TABLE_SUMMARY_VALUES])}")
    if not key_values and table_text_value:
        lines.append(f"内容概览：{clean_inline_text(table_text_value)[:1200]}")
    return "\n".join(line for line in lines if line).strip()


def table_row_chunk_content(*, source_lines: list[str], table: dict[str, Any], row_item: dict[str, Any]) -> str:
    """生成行级或键值级检索文本。"""
    lines = [
        "表格字段证据" if row_item["is_key_value"] else "表格行证据",
        *non_empty_lines(source_lines),
        table_identity_line(table),
        f"表格行号：第{row_item['row_index']}行",
    ]
    if row_item["is_key_value"]:
        lines.extend(
            [
                f"字段名称：{row_item['field_name']}",
                f"字段内容：{row_item['field_value']}",
            ]
        )
    lines.append(f"行内容：{row_item['text']}")
    return "\n".join(line for line in lines if line).strip()


def semantic_table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """把表格数据行转换为可索引的语义行。"""
    headers = table_headers(table)
    rows = table_rows(table)
    items = []
    for row_index, row in enumerate(rows, start=1):
        values = [clean_inline_text(str(value)) for value in list(row)]
        if not any(values):
            continue
        field_name, field_value = key_value_pair(headers, values)
        field_name = field_name[:MAX_TABLE_FIELD_NAME_CHARS]
        text = table_row_text(headers, values, row_index=row_index)
        items.append(
            {
                "row_index": row_index,
                "values": values,
                "text": clean_inline_text(text)[:MAX_TABLE_ROW_VALUE_CHARS],
                "is_key_value": bool(field_name and field_value),
                "field_name": field_name,
                "field_value": field_value[:MAX_TABLE_ROW_VALUE_CHARS] if field_value else "",
            }
        )
    return items


def key_value_pair(headers: list[str], values: list[str]) -> tuple[str, str]:
    """识别两列表或字段/内容式表格行。"""
    padded_values = [value for value in values if value]
    if len(padded_values) < 2:
        return "", ""
    normalized_headers = "".join(headers)
    looks_like_kv_header = any(token in normalized_headers for token in ["字段", "名称", "项目", "材料", "内容", "要求", "说明"])
    compact_row = len(padded_values) <= 3
    if looks_like_kv_header or compact_row:
        return padded_values[0], "；".join(padded_values[1:])
    return "", ""


def key_value_metadata(row_item: dict[str, Any]) -> dict[str, Any]:
    """返回键值行专用元数据。"""
    if not row_item["is_key_value"]:
        return {}
    return {
        "field_name": row_item["field_name"],
        "field_value": row_item["field_value"],
    }


def table_summary_text(table: dict[str, Any]) -> str:
    """表格文本缺失时生成兜底摘要。"""
    rows = semantic_table_rows(table)
    return "\n".join(item["text"] for item in rows[:MAX_TABLE_SUMMARY_VALUES]).strip()


def normalized_table_payload(value: Any) -> dict[str, Any]:
    """把外部表格 JSON 归一化为 dict。"""
    if not isinstance(value, dict):
        return {}
    headers = [clean_inline_text(str(item)) for item in list(value.get("headers") or [])]
    rows = []
    for row in list(value.get("rows") or []):
        if isinstance(row, list):
            rows.append([clean_inline_text(str(item)) for item in row])
    payload = dict(value)
    payload["headers"] = headers
    payload["rows"] = rows
    payload["row_count"] = int(value.get("row_count") or len(rows) + (1 if headers else 0))
    payload["column_count"] = int(value.get("column_count") or max([len(headers), *[len(row) for row in rows]], default=0))
    return payload


def table_content_dedupe_signature(table_json_value: Any, table_text_value: str = "") -> str:
    """返回忽略来源位置的表格内容去重签名。"""
    table = normalized_table_payload(table_json_value)
    if table:
        payload = {
            "title": normalize_table_dedupe_value(table.get("title")),
            "headers": [normalize_table_dedupe_value(item) for item in table_headers(table)],
            "rows": [
                [normalize_table_dedupe_value(item) for item in row]
                for row in table_rows(table)
                if any(normalize_table_dedupe_value(item) for item in row)
            ],
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    normalized_text = normalize_table_dedupe_value(table_text_value)
    return hashlib.sha1(normalized_text.encode("utf-8")).hexdigest() if normalized_text else ""


def normalize_table_dedupe_value(value: Any) -> str:
    """清理表格去重字段，避免空白差异造成重复入库。"""
    return " ".join(clean_inline_text(str(value or "")).split())


def table_headers(table: dict[str, Any]) -> list[str]:
    """返回表头列表。"""
    return [clean_inline_text(str(item)) for item in list(table.get("headers") or []) if clean_inline_text(str(item))]


def table_rows(table: dict[str, Any]) -> list[list[str]]:
    """返回表格数据行。"""
    rows = []
    for row in list(table.get("rows") or []):
        if isinstance(row, list):
            rows.append([clean_inline_text(str(item)) for item in row])
    return rows


def table_row_count(table: dict[str, Any]) -> int:
    """返回表格总行数。"""
    try:
        return int(table.get("row_count") or len(table_rows(table)))
    except (TypeError, ValueError):
        return len(table_rows(table))


def table_column_count(table: dict[str, Any]) -> int:
    """返回表格列数。"""
    try:
        return int(table.get("column_count") or len(table_headers(table)))
    except (TypeError, ValueError):
        return len(table_headers(table))


def table_identity_line(table: dict[str, Any]) -> str:
    """返回表格标题行。"""
    title = clean_inline_text(str(table.get("title") or ""))[:MAX_TABLE_TITLE_CHARS]
    return f"表格标题：{title}" if title else "表格标题：未命名表格"


def table_size_line(table: dict[str, Any]) -> str:
    """返回表格规模行。"""
    return f"表格规模：{table_row_count(table)} 行 x {table_column_count(table)} 列"


def table_semantic_id_prefix(*parts: Any) -> str:
    """构建稳定的表格语义 chunk 前缀。"""
    key = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"table_semantic:{digest}"


def non_empty_lines(lines: list[str]) -> list[str]:
    """清理来源上下文行。"""
    return [
        clean_inline_text(str(line))[:MAX_TABLE_SOURCE_LINE_CHARS]
        for line in lines
        if clean_inline_text(str(line))
    ]


def bounded_chunk_text(value: str) -> str:
    """限制发送到 embedding/rerank 的表格子块长度。"""
    return str(value or "")[:MAX_INDEXABLE_TABLE_CHUNK_CHARS].rstrip()


def coerce_table_index(value: Any, fallback: int = 0) -> int:
    """安全转换表格序号。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback or 0)
