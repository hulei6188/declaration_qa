"""申报材料文件的安全落盘与引用校验。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".jpg",
    ".jpeg",
    ".png",
    ".zip",
}


class ApplicationMaterialStore:
    """按用户和会话隔离保存申报材料，客户端只持有不可执行的附件引用。"""

    def __init__(self, root_dir: str | Path, *, max_bytes: int) -> None:
        self.root_dir = Path(root_dir)
        self.max_bytes = max(1, int(max_bytes or 1))

    def save(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str,
        user_id: int,
        session_id: str,
        item_code: str,
        field_name: str,
        material_key: str,
    ) -> dict[str, Any]:
        if not data:
            raise ValueError("申报材料不能为空")
        if len(data) > self.max_bytes:
            raise ValueError(f"单个申报材料不能超过 {self.max_bytes // (1024 * 1024)} MB")
        safe_filename = Path(str(filename or "material.bin")).name
        extension = Path(safe_filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError("仅支持 PDF、Word、Excel、JPG、PNG 或 ZIP 文件")
        safe_session = safe_segment(session_id, "session_id")
        safe_field = safe_segment(field_name, "field_name")
        safe_material = safe_segment(material_key, "material_key")
        attachment_id = uuid4().hex
        digest = hashlib.sha256(data).hexdigest()
        directory = self.root_dir / str(int(user_id)) / safe_session
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"{attachment_id}{extension}"
        metadata_path = directory / f"{attachment_id}.json"
        file_path.write_bytes(data)
        metadata = {
            "attachment_id": attachment_id,
            "filename": safe_filename,
            "content_type": str(content_type or "application/octet-stream")[:160],
            "size_bytes": len(data),
            "sha256": digest,
            "item_code": str(item_code or "")[:160],
            "field_name": safe_field,
            "material_key": safe_material,
            "extension": extension,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        return metadata

    def validate_reference(
        self,
        reference: dict[str, Any],
        *,
        user_id: int,
        session_id: str,
        field_name: str,
        material_key: str,
    ) -> dict[str, Any] | None:
        attachment_id = str(reference.get("attachment_id") or "")
        if not re.fullmatch(r"[a-f0-9]{32}", attachment_id):
            return None
        try:
            directory = self.root_dir / str(int(user_id)) / safe_segment(session_id, "session_id")
            metadata_path = directory / f"{attachment_id}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            extension = str(metadata.get("extension") or "")
            file_path = directory / f"{attachment_id}{extension}"
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if not file_path.is_file():
            return None
        if str(metadata.get("field_name") or "") != field_name:
            return None
        if str(metadata.get("material_key") or "") != material_key:
            return None
        if int(metadata.get("size_bytes") or 0) != file_path.stat().st_size:
            return None
        return metadata


def safe_segment(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", text):
        raise ValueError(f"无效的 {label}")
    return text
