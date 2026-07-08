"""附件下载和本地缓存。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from declaration_qa.indexing.attachments.models import AttachmentRef, DownloadedAttachment


class AttachmentDownloadError(RuntimeError):
    """附件下载失败。"""


@dataclass(frozen=True)
class AttachmentDownloadSettings:
    """附件下载安全配置。"""

    cache_dir: Path
    timeout_seconds: float = 10.0
    max_bytes: int = 20 * 1024 * 1024
    retries: int = 1
    user_agent: str = "declaration-qa-indexer/1.0"


class AttachmentDownloader:
    """带大小限制和缓存的附件下载器。"""

    def __init__(self, settings: AttachmentDownloadSettings) -> None:
        """初始化附件下载器。"""
        self.settings = settings

    def download(self, ref: AttachmentRef) -> DownloadedAttachment:
        """下载附件，已缓存时直接复用。"""
        validate_download_url(ref.url)
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_path(ref)
        if cache_path.exists():
            data = cache_path.read_bytes()
            return DownloadedAttachment(
                ref=ref,
                data=data,
                content_type=content_type_from_extension(ref.extension),
                cache_path=str(cache_path),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        last_error: Exception | None = None
        for _ in range(max(1, int(self.settings.retries) + 1)):
            try:
                data, content_type = self.fetch(ref)
                cache_path.write_bytes(data)
                return DownloadedAttachment(
                    ref=ref,
                    data=data,
                    content_type=content_type or content_type_from_extension(ref.extension),
                    cache_path=str(cache_path),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            except Exception as exc:
                last_error = exc
        raise AttachmentDownloadError(str(last_error or "download failed"))

    def fetch(self, ref: AttachmentRef) -> tuple[bytes, str]:
        """从远程地址流式下载附件并限制最大字节数。"""
        response = requests.get(
            ref.url,
            headers={"User-Agent": self.settings.user_agent, "Accept": "*/*"},
            timeout=self.settings.timeout_seconds,
            stream=True,
        )
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > self.settings.max_bytes:
            raise AttachmentDownloadError("attachment exceeds max size")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if not chunk:
                continue
            size += len(chunk)
            if size > self.settings.max_bytes:
                raise AttachmentDownloadError("attachment exceeds max size")
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            raise AttachmentDownloadError("attachment is empty")
        return data, str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip()

    def cache_path(self, ref: AttachmentRef) -> Path:
        """返回附件缓存路径。"""
        digest = hashlib.sha256(ref.url.encode("utf-8")).hexdigest()
        suffix = f".{ref.extension}" if ref.extension else ""
        return self.settings.cache_dir / f"{digest}{suffix}"


def validate_download_url(url: str) -> None:
    """限制附件下载仅支持 HTTP(S)。"""
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AttachmentDownloadError("unsupported attachment URL")


def content_type_from_extension(extension: str) -> str:
    """根据扩展名返回常见内容类型。"""
    ext = extension.lower().strip(".")
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }.get(ext, "application/octet-stream")

