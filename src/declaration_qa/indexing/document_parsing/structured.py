"""结构化解析器通用适配工具。"""

from __future__ import annotations

import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from declaration_qa.indexing.document_parsing.models import ParsedBlock, ParsedDocument
from declaration_qa.indexing.document_parsing.native import parse_metadata, table_blocks
from declaration_qa.indexing.document_parsing.text import normalize_extracted_text, normalized_table_rows


@contextmanager
def temp_document_file(*, filename: str, data: bytes) -> Iterator[Path]:
    """将上传字节写入临时文件，供第三方解析器读取。"""
    suffix = Path(filename or "document.bin").suffix or ".bin"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = Path(handle.name)
    try:
        with handle:
            handle.write(data)
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            return


def parsed_document_from_markdown(
    *,
    filename: str,
    content_type: str,
    parser_name: str,
    markdown: str,
    metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    """从 Markdown 文本构建标准解析结果。"""
    text = normalize_extracted_text(markdown)
    blocks = [ParsedBlock(block_type="text", text=text)] if text else []
    blocks.extend(markdown_table_blocks(text, parser_name=parser_name))
    metadata_payload = {**parse_metadata(parser_name, blocks), **(metadata or {})}
    return ParsedDocument(
        filename=filename,
        content_type=(content_type or "").lower(),
        text=text,
        parser_name=parser_name,
        blocks=blocks,
        metadata=metadata_payload,
    )


def markdown_table_blocks(markdown: str, *, parser_name: str) -> list[ParsedBlock]:
    """从 Markdown 中识别表格块并补充 HTML/JSON 表达。"""
    lines = str(markdown or "").splitlines()
    blocks: list[ParsedBlock] = []
    index = 0
    table_index = 0
    while index < len(lines):
        if not looks_like_markdown_table_start(lines, index):
            index += 1
            continue
        table_lines = [lines[index], lines[index + 1]]
        index += 2
        while index < len(lines) and lines[index].strip().startswith("|"):
            table_lines.append(lines[index])
            index += 1
        rows = markdown_rows_to_cells(table_lines)
        if rows:
            blocks.extend(table_blocks(rows, source=parser_name, table_index=table_index))
            table_index += 1
    return blocks


def looks_like_markdown_table_start(lines: list[str], index: int) -> bool:
    """判断当前位置是否像 Markdown 表格头。"""
    if index + 1 >= len(lines):
        return False
    head = lines[index].strip()
    separator = lines[index + 1].strip()
    return head.startswith("|") and head.endswith("|") and bool(re.match(r"^\|[\s:\-|]+\|$", separator))


def markdown_rows_to_cells(table_lines: list[str]) -> list[list[str]]:
    """将 Markdown 表格行转换为二维单元格。"""
    rows = []
    for line_index, line in enumerate(table_lines):
        if line_index == 1:
            continue
        stripped = line.strip().strip("|")
        cells = [cell.strip().replace("\\|", "|") for cell in stripped.split("|")]
        rows.append(cells)
    return normalized_table_rows(rows)


def metadata_summary(payload: Any) -> dict[str, Any]:
    """从第三方结构化输出中提取轻量诊断信息。"""
    if payload in (None, "", [], {}):
        return {}
    return {
        "parser_structured_output_type": type(payload).__name__,
        "parser_structured_output_chars": len(json.dumps(payload, ensure_ascii=False, default=str)[:20000]),
    }


def object_to_plain_data(value: Any) -> Any:
    """尽量把第三方对象转成普通 dict/list。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [object_to_plain_data(item) for item in list(value)]
    if isinstance(value, dict):
        return {str(key): object_to_plain_data(item) for key, item in value.items()}
    for method_name in ("model_dump", "dict", "export_to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return object_to_plain_data(method())
            except TypeError:
                try:
                    return object_to_plain_data(method(exclude_none=True))
                except Exception:
                    continue
            except Exception:
                continue
    return str(value)
