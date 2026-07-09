"""Docling 文档解析器适配。"""

from __future__ import annotations

from typing import Any

from declaration_qa.indexing.document_parsing.models import DocumentParseError, ParsedDocument
from declaration_qa.indexing.document_parsing.structured import (
    metadata_summary,
    object_to_plain_data,
    parsed_document_from_markdown,
    temp_document_file,
)


class DoclingDocumentParser:
    """基于 Docling 的本地结构化解析器。"""

    name = "docling"

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """调用 Docling 并转换为标准解析结果。"""
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise DocumentParseError("docling is not installed") from exc

        with temp_document_file(filename=filename, data=data) as path:
            try:
                result = DocumentConverter().convert(str(path))
                document = getattr(result, "document", result)
                markdown = export_document_markdown(document)
                structured_payload = object_to_plain_data(document)
            except Exception as exc:
                raise DocumentParseError(f"Docling parse failed: {type(exc).__name__}: {str(exc)[:240]}") from exc
        if not markdown.strip():
            raise DocumentParseError("Docling returned empty markdown")
        return parsed_document_from_markdown(
            filename=filename,
            content_type=content_type,
            parser_name=self.name,
            markdown=markdown,
            metadata={
                "parser_output_format": "markdown",
                **metadata_summary(structured_payload),
            },
        )


def export_document_markdown(document: Any) -> str:
    """从 Docling 文档对象导出 Markdown。"""
    for method_name in ("export_to_markdown", "export_to_text"):
        method = getattr(document, method_name, None)
        if callable(method):
            try:
                return str(method() or "")
            except Exception:
                continue
    return str(document or "")
