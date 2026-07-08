"""聊天图片查询载荷处理。"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from pathlib import Path

from fastapi import HTTPException, status

from declaration_qa.api.schemas import ChatImagePayload, ChatRequest
from declaration_qa.core.config import DeclarationQASettings


IMAGE_CONTENT_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}


def prepare_chat_image_request(request: ChatRequest, settings: DeclarationQASettings) -> ChatRequest:
    """将聊天图片 data URL 落盘，并移除请求中的 base64 内容。"""
    payload = request.query_image
    if payload is None or not payload.data_url:
        return request
    content_type, data = decode_image_data_url(payload.data_url, fallback_content_type=payload.content_type)
    if content_type not in IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported chat image type")
    max_bytes = max(1, int(settings.chat_image_max_bytes or 1))
    if len(data) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Chat image is too large")
    digest = hashlib.sha256(data).hexdigest()
    extension = IMAGE_CONTENT_TYPES[content_type]
    image_dir = Path(settings.chat_image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"{digest[:24]}.{extension}"
    if not path.exists():
        path.write_bytes(data)
    prepared_image = ChatImagePayload(
        filename=payload.filename or path.name,
        content_type=content_type,
        data_url=None,
        file_path=str(path),
        size_bytes=len(data),
        text_present=bool(payload.text_present),
    )
    return request.model_copy(update={"query_image": prepared_image})


def decode_image_data_url(data_url: str, *, fallback_content_type: str = "image/png") -> tuple[str, bytes]:
    """解析前端上传的图片 data URL。"""
    raw = str(data_url or "").strip()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chat image is empty")
    match = re.match(r"^data:([^;,]+);base64,(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        content_type = match.group(1).strip().lower()
        encoded = match.group(2)
    else:
        content_type = str(fallback_content_type or "image/png").strip().lower()
        encoded = raw
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chat image data") from exc
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chat image is empty")
    return content_type, data
