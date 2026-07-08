"""知识库附件入库模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttachmentRef:
    """Markdown 文档中解析出的附件引用。"""

    index: int
    name: str
    url: str
    extension: str
    section_path: str
    section_title: str
    node_title: str
    parent_section_id: str
    material_name: str = ""
    role: str = "other_attachment"


@dataclass(frozen=True)
class DownloadedAttachment:
    """已下载并缓存的附件内容。"""

    ref: AttachmentRef
    data: bytes
    content_type: str
    cache_path: str
    sha256: str


@dataclass(frozen=True)
class ExtractedAttachment:
    """附件抽取后的文本和结构化摘要。"""

    ref: AttachmentRef
    text: str
    content_type: str
    cache_path: str = ""
    sha256: str = ""
    status: str = "success"
    error_reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

