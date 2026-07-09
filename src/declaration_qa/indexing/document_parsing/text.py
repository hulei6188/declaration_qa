"""文档解析层文本清理工具。"""

from __future__ import annotations

import re
from html import escape, unescape
from typing import Any


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


def table_json(rows: list[list[str]]) -> dict[str, Any]:
    """将二维单元格转换为稳定 JSON 结构。"""
    return {
        "headers": rows[0] if rows else [],
        "rows": rows[1:] if len(rows) > 1 else [],
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
