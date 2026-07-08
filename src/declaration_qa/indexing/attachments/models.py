"""知识库附件入库模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class AttachmentAsset:
    """附件中可展示、可做多模态检索的视觉资产。"""

    asset_id: str
    asset_kind: str
    modality: str
    path: str
    content_type: str
    filename: str
    text: str = ""
    page_no: int | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    assets: list[AttachmentAsset] = field(default_factory=list)
