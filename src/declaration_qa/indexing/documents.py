"""上传文档解析层门面。"""

import requests

from declaration_qa.indexing.document_parsing import (
    DocumentParseError,
    HttpOcrClient,
    HttpVisionSummaryClient,
    ParsedBlock,
    ParsedDocument,
    build_document_parser,
    normalize_extracted_text,
    parse_document_bytes,
)
from declaration_qa.indexing.document_parsing.native import parse_docx as parse_docx_with_blocks
from declaration_qa.indexing.document_parsing.native import parse_pdf as parse_pdf_with_blocks


def parse_docx(data: bytes) -> str:
    """兼容旧调用方，仅返回 DOCX 提取文本。"""
    return parse_docx_with_blocks(data)[0]


def parse_pdf(data: bytes) -> str:
    """兼容旧调用方，仅返回 PDF 提取文本。"""
    return parse_pdf_with_blocks(data)[0]

__all__ = [
    "DocumentParseError",
    "HttpOcrClient",
    "HttpVisionSummaryClient",
    "ParsedBlock",
    "ParsedDocument",
    "build_document_parser",
    "normalize_extracted_text",
    "parse_document_bytes",
    "parse_docx",
    "parse_pdf",
]
