"""文档解析器工厂。"""

from __future__ import annotations

import logging

from declaration_qa.indexing.document_parsing.docling_parser import DoclingDocumentParser
from declaration_qa.indexing.document_parsing.llamaparse_parser import LlamaParseDocumentParser
from declaration_qa.indexing.document_parsing.marker_parser import MarkerDocumentParser
from declaration_qa.indexing.document_parsing.models import DocumentParseError, DocumentParser, ParsedDocument
from declaration_qa.indexing.document_parsing.native import NativeDocumentParser
from declaration_qa.indexing.document_parsing.ocr import HttpOcrClient


DEFAULT_AUTO_ORDER = "docling,marker,native"
LOGGER = logging.getLogger(__name__)


class CompositeDocumentParser:
    """按顺序尝试多个解析器。"""

    name = "composite"

    def __init__(self, parsers: list[DocumentParser]) -> None:
        """初始化组合解析器。"""
        self.parsers = parsers

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """依次调用解析器，直到成功解析。"""
        errors: list[str] = []
        for index, parser in enumerate(self.parsers):
            try:
                parsed = parser.parse(filename=filename, content_type=content_type, data=data)
                if errors:
                    metadata = {
                        **parsed.metadata,
                        "document_parser_attempt_errors": errors,
                    }
                    return ParsedDocument(
                        filename=parsed.filename,
                        content_type=parsed.content_type,
                        text=parsed.text,
                        parser_name=parsed.parser_name,
                        blocks=parsed.blocks,
                        metadata=metadata,
                    )
                return parsed
            except DocumentParseError as exc:
                errors.append(f"{parser.name}: {str(exc)}")
                next_parser = self.parsers[index + 1].name if index + 1 < len(self.parsers) else "none"
                LOGGER.warning(
                    "Document parser failed; trying fallback "
                    "filename=%r content_type=%r size_bytes=%d parser=%s next_parser=%s error=%s",
                    filename,
                    content_type,
                    len(data),
                    parser.name,
                    next_parser,
                    exc,
                    exc_info=True,
                )
        raise DocumentParseError("; ".join(errors) if errors else "No document parser configured")


def parse_document_bytes(
    *,
    filename: str,
    content_type: str,
    data: bytes,
    ocr_client: HttpOcrClient | None = None,
    parser_provider: str = "auto",
    parser_order: str = "",
    parser_fallback_enabled: bool = True,
    llamaparse_api_key: str = "",
    llamaparse_base_url: str = "",
    llamaparse_result_type: str = "markdown",
    docling_document_timeout_seconds: int = 3600,
    docling_accelerator_device: str = "auto",
    docling_accelerator_num_threads: int = 4,
    docling_layout_batch_size: int = 4,
    docling_table_batch_size: int = 4,
    docling_ocr_batch_size: int = 4,
) -> ParsedDocument:
    """根据配置解析上传文档字节。"""
    effective_order = parser_order
    if normalize_provider(parser_provider) == "auto" and ocr_client and is_image_document(filename, content_type):
        effective_order = native_first_parser_order(parser_order)
    parser = build_document_parser(
        provider=parser_provider,
        order=effective_order,
        fallback_enabled=parser_fallback_enabled,
        ocr_client=ocr_client,
        llamaparse_api_key=llamaparse_api_key,
        llamaparse_base_url=llamaparse_base_url,
        llamaparse_result_type=llamaparse_result_type,
        docling_document_timeout_seconds=docling_document_timeout_seconds,
        docling_accelerator_device=docling_accelerator_device,
        docling_accelerator_num_threads=docling_accelerator_num_threads,
        docling_layout_batch_size=docling_layout_batch_size,
        docling_table_batch_size=docling_table_batch_size,
        docling_ocr_batch_size=docling_ocr_batch_size,
    )
    return parser.parse(filename=filename, content_type=content_type, data=data)


def build_document_parser(
    *,
    provider: str,
    order: str = "",
    fallback_enabled: bool = True,
    ocr_client: HttpOcrClient | None = None,
    llamaparse_api_key: str = "",
    llamaparse_base_url: str = "",
    llamaparse_result_type: str = "markdown",
    docling_document_timeout_seconds: int = 3600,
    docling_accelerator_device: str = "auto",
    docling_accelerator_num_threads: int = 4,
    docling_layout_batch_size: int = 4,
    docling_table_batch_size: int = 4,
    docling_ocr_batch_size: int = 4,
) -> DocumentParser:
    """根据 provider 构建文档解析器。"""
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "auto":
        names = parser_order(order or DEFAULT_AUTO_ORDER)
        return CompositeDocumentParser(
            [
                parser_by_name(
                    name,
                    ocr_client=ocr_client,
                    llamaparse_api_key=llamaparse_api_key,
                    llamaparse_base_url=llamaparse_base_url,
                    llamaparse_result_type=llamaparse_result_type,
                    docling_document_timeout_seconds=docling_document_timeout_seconds,
                    docling_accelerator_device=docling_accelerator_device,
                    docling_accelerator_num_threads=docling_accelerator_num_threads,
                    docling_layout_batch_size=docling_layout_batch_size,
                    docling_table_batch_size=docling_table_batch_size,
                    docling_ocr_batch_size=docling_ocr_batch_size,
                )
                for name in names
            ]
        )
    parser = parser_by_name(
        normalized_provider,
        ocr_client=ocr_client,
        llamaparse_api_key=llamaparse_api_key,
        llamaparse_base_url=llamaparse_base_url,
        llamaparse_result_type=llamaparse_result_type,
        docling_document_timeout_seconds=docling_document_timeout_seconds,
        docling_accelerator_device=docling_accelerator_device,
        docling_accelerator_num_threads=docling_accelerator_num_threads,
        docling_layout_batch_size=docling_layout_batch_size,
        docling_table_batch_size=docling_table_batch_size,
        docling_ocr_batch_size=docling_ocr_batch_size,
    )
    if normalized_provider == "native" or not fallback_enabled:
        return parser
    return CompositeDocumentParser([parser, NativeDocumentParser(ocr_client=ocr_client)])


def parser_by_name(
    name: str,
    *,
    ocr_client: HttpOcrClient | None,
    llamaparse_api_key: str,
    llamaparse_base_url: str,
    llamaparse_result_type: str,
    docling_document_timeout_seconds: int = 3600,
    docling_accelerator_device: str = "auto",
    docling_accelerator_num_threads: int = 4,
    docling_layout_batch_size: int = 4,
    docling_table_batch_size: int = 4,
    docling_ocr_batch_size: int = 4,
) -> DocumentParser:
    """根据名称返回解析器实例。"""
    normalized = normalize_provider(name)
    if normalized == "docling":
        return DoclingDocumentParser(
            document_timeout_seconds=docling_document_timeout_seconds,
            accelerator_device=docling_accelerator_device,
            accelerator_num_threads=docling_accelerator_num_threads,
            layout_batch_size=docling_layout_batch_size,
            table_batch_size=docling_table_batch_size,
            ocr_batch_size=docling_ocr_batch_size,
        )
    if normalized == "marker":
        return MarkerDocumentParser()
    if normalized == "llamaparse":
        return LlamaParseDocumentParser(
            api_key=llamaparse_api_key,
            base_url=llamaparse_base_url,
            result_type=llamaparse_result_type,
        )
    if normalized == "native":
        return NativeDocumentParser(ocr_client=ocr_client)
    raise DocumentParseError(f"Unsupported document parser provider: {name}")


def parser_order(value: str) -> list[str]:
    """解析逗号分隔的解析器顺序。"""
    names = [normalize_provider(item) for item in str(value or "").split(",") if normalize_provider(item)]
    return names or parser_order(DEFAULT_AUTO_ORDER)


def native_first_parser_order(value: str) -> str:
    """图片已配置项目 OCR 时优先使用 native 解析器。"""
    names = parser_order(value or DEFAULT_AUTO_ORDER)
    return ",".join(["native", *(name for name in names if name != "native")])


def is_image_document(filename: str, content_type: str) -> bool:
    """判断输入是否为独立图片文件。"""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return suffix in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"} or str(content_type or "").lower().startswith("image/")


def normalize_provider(value: str) -> str:
    """标准化解析器名称。"""
    normalized = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "llama-parse": "llamaparse",
        "llama_parse": "llamaparse",
        "llama": "llamaparse",
        "built-in": "native",
        "builtin": "native",
        "current": "native",
    }
    return aliases.get(normalized, normalized)
