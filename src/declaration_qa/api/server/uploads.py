"""知识文件上传暂存和索引任务创建。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from fastapi import HTTPException, UploadFile

from declaration_qa.api.schemas import KnowledgeUploadResponse
from declaration_qa.rag.engine.chunking import normalize_chunking_strategy
from declaration_qa.rag.retrieval import sanitize_metadata

def parse_metadata_form(value: str) -> dict:
    """解析表单中的元数据 JSON。"""
    if not str(value or "").strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return sanitize_metadata(parsed)


def enqueue_upload_document(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    title: str,
    region_code: str,
    policy_version: str,
    source_url: str,
    audit_node: str,
    status: str,
    effective_start_time: str,
    effective_end_time: str,
    metadata: dict,
    chunking_strategy: str,
) -> KnowledgeUploadResponse:
    """创建上传文档索引任务并投递队列。"""
    from declaration_qa.api.server.runtime import index_job_store, index_queue

    document_id = stable_upload_document_id(data)
    staged_path = stage_upload_bytes(data, filename=filename, document_id=document_id)
    strategy = normalize_chunking_strategy(chunking_strategy)
    job = index_job_store.create_index_job(
        job_type="document_index",
        payload={
            "input_type": "file",
            "document_id": document_id,
            "title": title or filename or document_id,
            "filename": filename or staged_path.name,
            "content_type": content_type or "application/octet-stream",
            "file_path": str(staged_path),
            "chunking_strategy": strategy,
            "region_code": region_code,
            "policy_version": policy_version,
            "source_url": source_url,
            "audit_node": audit_node,
            "status": status,
            "effective_start_time": effective_start_time or None,
            "effective_end_time": effective_end_time or None,
            "metadata": sanitize_metadata(metadata),
        },
    )
    index_queue.enqueue(job["job_uid"])
    final_job = index_job_store.get_index_job(job["job_uid"]) or job
    final_payload = final_job.get("payload") or {}
    return KnowledgeUploadResponse(
        document_id=document_id,
        chunks_indexed=int(final_payload.get("chunks_indexed") or 0),
        job_id=job["job_uid"],
        filename=str(final_payload.get("filename") or filename or staged_path.name),
        content_type=str(final_payload.get("content_type") or content_type or "application/octet-stream"),
        chunking_strategy=str(final_payload.get("chunking_strategy") or strategy),
    )


def batch_upload_title(title: str, filename: str, file_count: int) -> str:
    """生成批量上传文档标题。"""
    if not title:
        return filename
    return f"{title} - {filename}" if file_count > 1 else title

async def stage_upload_file(file: UploadFile, *, document_id: str) -> Path:
    """将上传文件保存到临时目录。"""
    return stage_upload_bytes(await file.read(), filename=file.filename or "upload.bin", document_id=document_id)


def stage_upload_bytes(data: bytes, *, filename: str, document_id: str) -> Path:
    """将文件字节写入上传暂存目录。"""
    from declaration_qa.api.server.runtime import settings

    settings.effective_upload_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(filename)
    path = settings.effective_upload_dir / f"{document_id}-{filename}"
    path.write_bytes(data)
    return path


def stable_upload_document_id(data: bytes) -> str:
    """根据文件名和内容生成稳定文档 ID。"""
    return f"doc-{hashlib.sha256(data).hexdigest()[:32]}"


def safe_filename(value: str) -> str:
    """清理上传文件名中的不安全字符。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "upload.bin"
