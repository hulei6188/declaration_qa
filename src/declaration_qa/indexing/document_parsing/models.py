"""文档解析层标准模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class DocumentParseError(ValueError):
    """文档解析失败时抛出的业务异常。"""


@dataclass(frozen=True)
class ParsedBlock:
    """解析器输出的标准内容块。"""

    block_type: str
    text: str
    html: str = ""
    json_value: Any = None
    page_no: int | None = None
    bbox: list[float] | None = None
    binary_data: bytes = b""
    content_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """上传文档解析后的标准结果。"""

    filename: str
    content_type: str
    text: str
    parser_name: str = "native"
    blocks: list[ParsedBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(Protocol):
    """文档解析器协议。"""

    name: str

    def parse(
        self,
        *,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> ParsedDocument:
        """解析文件字节并返回标准文档结果。"""
