"""LlamaParse 文档解析器适配。"""

from __future__ import annotations

from declaration_qa.indexing.document_parsing.models import DocumentParseError, ParsedDocument
from declaration_qa.indexing.document_parsing.structured import parsed_document_from_markdown, temp_document_file


class LlamaParseDocumentParser:
    """基于 LlamaParse 的云端结构化解析器。"""

    name = "llamaparse"

    def __init__(self, *, api_key: str = "", base_url: str = "", result_type: str = "markdown") -> None:
        """初始化 LlamaParse 解析器。"""
        self.api_key = api_key
        self.base_url = base_url
        self.result_type = result_type or "markdown"

    def parse(self, *, filename: str, content_type: str, data: bytes) -> ParsedDocument:
        """调用 LlamaParse 并转换为标准解析结果。"""
        if not self.api_key:
            raise DocumentParseError("DECLARATION_QA_LLAMAPARSE_API_KEY is required for LlamaParse")
        LlamaParse = import_llamaparse_class()
        with temp_document_file(filename=filename, data=data) as path:
            try:
                kwargs = {
                    "api_key": self.api_key,
                    "result_type": self.result_type,
                }
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                parser = LlamaParse(**kwargs)
                documents = parser.load_data(str(path))
                markdown = "\n\n".join(str(getattr(item, "text", "") or "") for item in documents)
            except Exception as exc:
                raise DocumentParseError(f"LlamaParse parse failed: {type(exc).__name__}: {str(exc)[:240]}") from exc
        if not markdown.strip():
            raise DocumentParseError("LlamaParse returned empty text")
        return parsed_document_from_markdown(
            filename=filename,
            content_type=content_type,
            parser_name=self.name,
            markdown=markdown,
            metadata={"parser_output_format": self.result_type},
        )


def import_llamaparse_class():
    """兼容 LlamaParse 新旧 Python 包名。"""
    try:
        from llama_cloud_services import LlamaParse

        return LlamaParse
    except ImportError:
        pass
    try:
        from llama_parse import LlamaParse

        return LlamaParse
    except ImportError as exc:
        raise DocumentParseError("llama-cloud-services or llama-parse is not installed") from exc
