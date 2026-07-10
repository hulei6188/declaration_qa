"""文档解析层公共入口。"""

from declaration_qa.indexing.document_parsing.factory import build_document_parser, parse_document_bytes
from declaration_qa.indexing.document_parsing.models import DocumentParseError, ParsedBlock, ParsedDocument
from declaration_qa.indexing.document_parsing.ocr import HttpOcrClient
from declaration_qa.indexing.document_parsing.text import normalize_extracted_text
from declaration_qa.indexing.document_parsing.vision_summary import HttpVisionSummaryClient

__all__ = [
    "DocumentParseError",
    "HttpOcrClient",
    "HttpVisionSummaryClient",
    "ParsedBlock",
    "ParsedDocument",
    "build_document_parser",
    "normalize_extracted_text",
    "parse_document_bytes",
]
