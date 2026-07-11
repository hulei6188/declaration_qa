"""会话消息载荷整理工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def user_message_payload(*, content: str, created_at: str = "", query_image: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建历史会话中的用户消息。"""
    attachments = query_image_attachments(query_image)
    payload: dict[str, Any] = {
        "role": "user",
        "content": content,
        "created_at": created_at,
    }
    if attachments:
        payload["attachments"] = attachments
    return payload


def query_image_attachments(query_image: dict[str, Any] | None) -> list[dict[str, Any]]:
    """将落盘后的聊天图片转换为前端附件。"""
    if not isinstance(query_image, dict) or not query_image:
        return []
    file_path = str(query_image.get("file_path") or "").strip()
    filename = str(query_image.get("filename") or "").strip()
    image_name = Path(file_path).name if file_path else filename
    if not image_name:
        return []
    return [
        {
            "filename": filename or image_name,
            "content_type": str(query_image.get("content_type") or "image/png"),
            "size_bytes": int(query_image.get("size_bytes") or 0),
            "url": f"/api/declaration-qa/chat-images/{image_name}",
        }
    ]


def normalize_message_row(row: Any) -> dict[str, Any]:
    """将数据库消息行转换为会话接口消息。"""
    payload = dict(row)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    message = {
        "role": str(payload.get("role") or ""),
        "content": str(payload.get("content") or ""),
        "intent": payload.get("intent") or "",
        "refused": bool(payload.get("refused")),
        "refuse_reason": payload.get("refuse_reason") or "",
        "sources": payload.get("sources") or [],
        "suggested_actions": payload.get("suggested_actions") or [],
        "meta": meta,
        "run_id": str(payload.get("run_id") or ""),
        "created_at": str(payload.get("created_at") or ""),
    }
    if message["role"] == "user":
        attachments = meta.get("attachments") or query_image_attachments(meta.get("query_image"))
        if attachments:
            message["attachments"] = attachments
    return message
