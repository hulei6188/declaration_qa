"""文档解析器工厂。"""

from __future__ import annotations

import logging

from gov_service_agent.indexing.document_parsing.llamaparse_parser import LlamaParseDocumentParser
from gov_service_agent.indexing.document_parsing.legacy_doc import (
    convert_legacy_doc_to_docx,
    is_legacy_word_document,
)
from gov_service_agent.indexing.document_parsing.marker_parser import MarkerDocumentParser
from gov_service_agent.indexing.document_parsing.mineru_parser import MinerUDocumentParser
from gov_service_agent.indexing.document_parsing.models import DocumentParseError, DocumentParser, ParsedDocument
from gov_service_agent.indexing.document_parsing.native import NativeDocumentParser
from gov_service_agent.indexing.document_parsing.ocr import HttpOcrClient


DEFAULT_AUTO_ORDER = "mineru,native"
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
            supports = getattr(parser, "supports", None)
            if callable(supports) and not supports(filename=filename, content_type=content_type):
                continue
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


class LegacyDocConvertingDocumentParser:
    """先将旧版 .doc 文件转换为 .docx，再交给已配置的解析器处理。"""

    name = "legacy-doc-converter"

    def __init__(
        self,
        parser: DocumentParser,
        *,
        converter_command: str = "",
        timeout_seconds: int = 120,
    ) -> None:
        """初始化格式转换包装器。"""
        self.parser = parser
        self.converter_command = converter_command
        self.timeout_seconds = timeout_seconds

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """先转换旧版 Word 文件，再解析生成的 DOCX。"""
        if not is_legacy_word_document(filename, content_type):
            return self.parser.parse(filename=filename, content_type=content_type, data=data)
        converted = convert_legacy_doc_to_docx(
            filename=filename,
            data=data,
            converter_command=self.converter_command,
            timeout_seconds=self.timeout_seconds,
        )
        parsed = self.parser.parse(
            filename=converted.filename,
            content_type=converted.content_type,
            data=converted.data,
        )
        metadata = {
            **parsed.metadata,
            "legacy_doc_converted": True,
            "legacy_doc_converter": "libreoffice",
            "legacy_doc_converter_command": converted.converter_command,
            "legacy_doc_original_filename": filename,
            "legacy_doc_original_content_type": content_type,
            "legacy_doc_converted_filename": converted.filename,
            "legacy_doc_converted_content_type": converted.content_type,
        }
        return ParsedDocument(
            filename=filename,
            content_type=(content_type or "").lower(),
            text=parsed.text,
            parser_name=parsed.parser_name,
            blocks=parsed.blocks,
            metadata=metadata,
        )


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
    mineru_api_url: str = "http://127.0.0.1:8011",
    mineru_backend: str = "pipeline",
    mineru_method: str = "auto",
    mineru_language: str = "ch",
    mineru_model_source: str = "modelscope",
    mineru_formula_enabled: bool = False,
    mineru_timeout_seconds: int = 3600,
    chonkie_text_max_chars: int = 900,
    chonkie_text_min_chars: int = 120,
    chonkie_table_max_chars: int = 3500,
    legacy_doc_conversion_enabled: bool = True,
    legacy_doc_converter_command: str = "",
    legacy_doc_conversion_timeout_seconds: int = 120,
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
        mineru_api_url=mineru_api_url,
        mineru_backend=mineru_backend,
        mineru_method=mineru_method,
        mineru_language=mineru_language,
        mineru_model_source=mineru_model_source,
        mineru_formula_enabled=mineru_formula_enabled,
        mineru_timeout_seconds=mineru_timeout_seconds,
        chonkie_text_max_chars=chonkie_text_max_chars,
        chonkie_text_min_chars=chonkie_text_min_chars,
        chonkie_table_max_chars=chonkie_table_max_chars,
        legacy_doc_conversion_enabled=legacy_doc_conversion_enabled,
        legacy_doc_converter_command=legacy_doc_converter_command,
        legacy_doc_conversion_timeout_seconds=legacy_doc_conversion_timeout_seconds,
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
    mineru_api_url: str = "http://127.0.0.1:8011",
    mineru_backend: str = "pipeline",
    mineru_method: str = "auto",
    mineru_language: str = "ch",
    mineru_model_source: str = "modelscope",
    mineru_formula_enabled: bool = False,
    mineru_timeout_seconds: int = 3600,
    chonkie_text_max_chars: int = 900,
    chonkie_text_min_chars: int = 120,
    chonkie_table_max_chars: int = 3500,
    legacy_doc_conversion_enabled: bool = False,
    legacy_doc_converter_command: str = "",
    legacy_doc_conversion_timeout_seconds: int = 120,
) -> DocumentParser:
    """根据 provider 构建文档解析器。"""
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "auto":
        names = parser_order(order or DEFAULT_AUTO_ORDER)
        parser = CompositeDocumentParser(
            [
                parser_by_name(
                    name,
                    ocr_client=ocr_client,
                    llamaparse_api_key=llamaparse_api_key,
                    llamaparse_base_url=llamaparse_base_url,
                    llamaparse_result_type=llamaparse_result_type,
                    mineru_api_url=mineru_api_url,
                    mineru_backend=mineru_backend,
                    mineru_method=mineru_method,
                    mineru_language=mineru_language,
                    mineru_model_source=mineru_model_source,
                    mineru_formula_enabled=mineru_formula_enabled,
                    mineru_timeout_seconds=mineru_timeout_seconds,
                    chonkie_text_max_chars=chonkie_text_max_chars,
                    chonkie_text_min_chars=chonkie_text_min_chars,
                    chonkie_table_max_chars=chonkie_table_max_chars,
                )
                for name in names
            ]
        )
        return wrap_legacy_doc_converter(
            parser,
            enabled=legacy_doc_conversion_enabled,
            converter_command=legacy_doc_converter_command,
            timeout_seconds=legacy_doc_conversion_timeout_seconds,
        )
    parser = parser_by_name(
        normalized_provider,
        ocr_client=ocr_client,
        llamaparse_api_key=llamaparse_api_key,
        llamaparse_base_url=llamaparse_base_url,
        llamaparse_result_type=llamaparse_result_type,
        mineru_api_url=mineru_api_url,
        mineru_backend=mineru_backend,
        mineru_method=mineru_method,
        mineru_language=mineru_language,
        mineru_model_source=mineru_model_source,
        mineru_formula_enabled=mineru_formula_enabled,
        mineru_timeout_seconds=mineru_timeout_seconds,
        chonkie_text_max_chars=chonkie_text_max_chars,
        chonkie_text_min_chars=chonkie_text_min_chars,
        chonkie_table_max_chars=chonkie_table_max_chars,
    )
    if normalized_provider == "native" or not fallback_enabled:
        return wrap_legacy_doc_converter(
            parser,
            enabled=legacy_doc_conversion_enabled,
            converter_command=legacy_doc_converter_command,
            timeout_seconds=legacy_doc_conversion_timeout_seconds,
        )
    return wrap_legacy_doc_converter(
        CompositeDocumentParser([parser, NativeDocumentParser(ocr_client=ocr_client)]),
        enabled=legacy_doc_conversion_enabled,
        converter_command=legacy_doc_converter_command,
        timeout_seconds=legacy_doc_conversion_timeout_seconds,
    )


def wrap_legacy_doc_converter(
    parser: DocumentParser,
    *,
    enabled: bool,
    converter_command: str,
    timeout_seconds: int,
) -> DocumentParser:
    """按需为解析器包装旧版 DOC 转换能力。"""
    if not enabled:
        return parser
    return LegacyDocConvertingDocumentParser(
        parser,
        converter_command=converter_command,
        timeout_seconds=timeout_seconds,
    )


def parser_by_name(
    name: str,
    *,
    ocr_client: HttpOcrClient | None,
    llamaparse_api_key: str,
    llamaparse_base_url: str,
    llamaparse_result_type: str,
    mineru_api_url: str = "http://127.0.0.1:8011",
    mineru_backend: str = "pipeline",
    mineru_method: str = "auto",
    mineru_language: str = "ch",
    mineru_model_source: str = "modelscope",
    mineru_formula_enabled: bool = False,
    mineru_timeout_seconds: int = 3600,
    chonkie_text_max_chars: int = 900,
    chonkie_text_min_chars: int = 120,
    chonkie_table_max_chars: int = 3500,
) -> DocumentParser:
    """根据名称返回解析器实例。"""
    normalized = normalize_provider(name)
    if normalized == "mineru":
        return MinerUDocumentParser(
            api_url=mineru_api_url,
            backend=mineru_backend,
            method=mineru_method,
            language=mineru_language,
            model_source=mineru_model_source,
            formula_enabled=mineru_formula_enabled,
            timeout_seconds=mineru_timeout_seconds,
            chonkie_text_max_chars=chonkie_text_max_chars,
            chonkie_text_min_chars=chonkie_text_min_chars,
            chonkie_table_max_chars=chonkie_table_max_chars,
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
        "magic-pdf": "mineru",
        "magicpdf": "mineru",
        "llama-parse": "llamaparse",
        "llama_parse": "llamaparse",
        "llama": "llamaparse",
        "built-in": "native",
        "builtin": "native",
        "current": "native",
    }
    return aliases.get(normalized, normalized)
