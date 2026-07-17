"""Marker 文档解析器适配。"""

from __future__ import annotations

from gov_service_agent.indexing.document_parsing.models import DocumentParseError, ParsedDocument
from gov_service_agent.indexing.document_parsing.structured import (
    metadata_summary,
    object_to_plain_data,
    parsed_document_from_markdown,
    temp_document_file,
)


class MarkerDocumentParser:
    """基于 Marker 的 PDF/图片结构化解析器。"""

    name = "marker"

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """调用 Marker 并转换为标准解析结果。"""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError as exc:
            raise DocumentParseError("marker-pdf is not installed") from exc

        with temp_document_file(filename=filename, data=data) as path:
            try:
                converter = PdfConverter(artifact_dict=create_model_dict())
                rendered = converter(str(path))
                markdown, _, _ = text_from_rendered(rendered)
                structured_payload = object_to_plain_data(rendered)
            except Exception as exc:
                raise DocumentParseError(f"Marker parse failed: {type(exc).__name__}: {str(exc)[:240]}") from exc
        if not str(markdown or "").strip():
            raise DocumentParseError("Marker returned empty markdown")
        return parsed_document_from_markdown(
            filename=filename,
            content_type=content_type,
            parser_name=self.name,
            markdown=str(markdown),
            metadata={
                "parser_output_format": "markdown",
                **metadata_summary(structured_payload),
            },
        )
