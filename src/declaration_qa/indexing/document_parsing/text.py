"""文档解析层文本清理工具。"""

from __future__ import annotations

import re
from html import escape, unescape
from typing import Any

MAX_TABLE_TEXT_ROWS = 80
MAX_COMPACT_TABLE_ROWS = 30
MAX_COMPACT_ROW_CELLS = 12
MAX_TABLE_JSON_CELLS = 1000


def decode_text(data: bytes) -> str:
    """按常见编码解码文本文件内容。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def normalize_extracted_text(text: str) -> str:
    """清理并压缩抽取文本中的空白字符。"""
    normalized = unescape(str(text or ""))
    normalized = re.sub(r"[ \t\r\f\v]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def clean_inline_text(text: str) -> str:
    """清理段落内多余空白。"""
    normalized = re.sub(r"[ \t\r\f\v]+", " ", text or "")
    normalized = re.sub(r"\n+", " ", normalized)
    return normalized.strip()


def escape_table_cell(text: str) -> str:
    """转义 Markdown 表格单元格。"""
    return clean_inline_text(text).replace("|", "\\|")


def markdown_table_row(values: list[str]) -> str:
    """拼接 Markdown 表格行。"""
    return "| " + " | ".join(values) + " |"


def markdown_table(rows: list[list[str]]) -> str:
    """将二维单元格转换为 Markdown 表格。"""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [markdown_table_row(padded_rows[0])]
    lines.append(markdown_table_row(["---"] * width))
    lines.extend(markdown_table_row(row) for row in padded_rows[1:])
    return "\n".join(lines)


def html_table(rows: list[list[str]]) -> str:
    """将二维单元格转换为 HTML 表格。"""
    if not rows:
        return ""
    lines = ["<table>"]
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 else "td"
        cells = "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row)
        lines.append(f"<tr>{cells}</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def table_json(
    rows: list[list[str]],
    *,
    cells: list[dict[str, Any]] | None = None,
    title: str = "",
    page_no: int | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    """将二维单元格转换为稳定 JSON 结构。"""
    normalized_rows = normalized_table_rows(rows)
    payload: dict[str, Any] = {
        "headers": normalized_rows[0] if normalized_rows else [],
        "rows": normalized_rows[1:] if len(normalized_rows) > 1 else [],
        "row_count": len(normalized_rows),
        "column_count": max((len(row) for row in normalized_rows), default=0),
    }
    if title:
        payload["title"] = clean_inline_text(title)
    if page_no is not None:
        payload["page_no"] = page_no
    if bbox is not None:
        payload["bbox"] = bbox
    if cells:
        payload["cells"] = cells[:MAX_TABLE_JSON_CELLS]
        payload["cells_truncated"] = len(cells) > MAX_TABLE_JSON_CELLS
    return payload


def table_text(rows: list[list[str]], *, title: str = "", max_rows: int = MAX_TABLE_TEXT_ROWS) -> str:
    """将表格转换成适合 embedding/BM25/rerank 的自然语言文本。"""
    normalized_rows = normalized_table_rows(rows)
    if not normalized_rows:
        return ""
    headers = normalized_rows[0]
    data_rows = normalized_rows[1:] if len(normalized_rows) > 1 else []
    lines = []
    if title:
        lines.append(f"结构化表格：{clean_inline_text(title)}")
    else:
        lines.append("结构化表格")
    if headers:
        lines.append(f"表头：{'、'.join(cell for cell in headers if cell)}")
    visible_rows = data_rows[:max_rows]
    for row_index, row in enumerate(visible_rows, start=1):
        lines.append(table_row_text(headers, row, row_index=row_index))
    if len(data_rows) > len(visible_rows):
        lines.append(f"表格还有 {len(data_rows) - len(visible_rows)} 行未展开。")
    return "\n".join(line for line in lines if line).strip()


def table_row_text(headers: list[str], row: list[str], *, row_index: int) -> str:
    """将单行表格转为自然语言。"""
    if headers and len(headers) == len(row):
        pairs = [
            f"{header or f'列{column_index + 1}'}：{cell}"
            for column_index, (header, cell) in enumerate(zip(headers, row))
            if str(cell or "").strip()
        ]
        return f"第{row_index}行：" + "；".join(pairs)
    values = [cell for cell in row if str(cell or "").strip()]
    return f"第{row_index}行：" + " | ".join(values)


def compact_table_context(
    table: dict[str, Any] | list[list[str]] | None,
    *,
    max_rows: int = MAX_COMPACT_TABLE_ROWS,
    max_cells_per_row: int = MAX_COMPACT_ROW_CELLS,
) -> dict[str, Any]:
    """生成适合放入 LLM prompt 的紧凑表格结构。"""
    if table is None:
        return {}
    if isinstance(table, list):
        source = table_json(table)
    elif isinstance(table, dict):
        source = dict(table)
    else:
        return {}
    headers = [clean_inline_text(str(item)) for item in list(source.get("headers") or [])]
    rows = source.get("rows") or []
    compact_rows = []
    for row_index, row in enumerate(list(rows)[:max_rows], start=1):
        row_cells = [clean_inline_text(str(item)) for item in list(row)[:max_cells_per_row]]
        compact_rows.append({"row": row_index, "cells": row_cells})
    return {
        "title": source.get("title") or "",
        "headers": headers[:max_cells_per_row],
        "rows": compact_rows,
        "row_count": int(source.get("row_count") or len(rows) + (1 if headers else 0)),
        "column_count": int(source.get("column_count") or len(headers)),
        "truncated": len(rows) > max_rows or any(len(list(row)) > max_cells_per_row for row in rows if isinstance(row, list)),
    }


def normalized_table_rows(rows: list[list[Any]]) -> list[list[str]]:
    """清理表格行列空白，同时保留行内列位置。"""
    cleaned_rows = []
    for row in rows:
        cells = [escape_table_cell(str(cell).replace("\xa0", " ")) if cell is not None else "" for cell in row]
        if any(cells):
            cleaned_rows.append(cells)
    if not cleaned_rows:
        return []
    width = max(last_non_empty_index(row) + 1 for row in cleaned_rows)
    return [row[:width] + [""] * max(0, width - len(row)) for row in cleaned_rows]


def last_non_empty_index(row: list[str]) -> int:
    """返回最后一个非空单元格下标。"""
    for index in range(len(row) - 1, -1, -1):
        if row[index].strip():
            return index
    return -1
