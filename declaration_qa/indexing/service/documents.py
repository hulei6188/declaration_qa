"""单篇知识文档索引处理。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.core.domain import KnowledgeDocument
from declaration_qa.indexing.documents import HttpOcrClient, parse_document_bytes
from declaration_qa.indexing.service.metadata import extract_document_metadata
from declaration_qa.indexing.service.workflow_index import index_document_for_workflow
from declaration_qa.rag.engine.chunking import normalize_chunking_strategy
from declaration_qa.rag.retrieval import sanitize_metadata

if TYPE_CHECKING:
    from declaration_qa.application.workflow import DeclarationQAWorkflow


def process_document_index(
    payload: dict[str, Any],
    *,
    workflow: DeclarationQAWorkflow,
    settings: DeclarationQASettings,
) -> dict[str, Any]:
    """处理单篇知识文档的索引入库。"""
    input_type = str(payload.get("input_type") or "text")
    if input_type == "file":
        file_path = Path(str(payload.get("file_path") or ""))
        if not file_path.exists():
            raise FileNotFoundError(f"Uploaded file is missing: {file_path}")
        data = file_path.read_bytes()
        ocr_client = (
            HttpOcrClient(url=settings.ocr_api_url, api_key=settings.ocr_api_key, model=settings.ocr_model)
            if settings.ocr_api_url
            else None
        )
        parsed = parse_document_bytes(
            filename=str(payload.get("filename") or file_path.name),
            content_type=str(payload.get("content_type") or "application/octet-stream"),
            data=data,
            ocr_client=ocr_client,
        )
        content = parsed.text
        filename = parsed.filename
        content_type = parsed.content_type
    else:
        content = str(payload.get("content") or "")
        filename = str(payload.get("filename") or "")
        content_type = str(payload.get("content_type") or "text/plain")
    title = str(payload.get("title") or filename or payload["document_id"])
    source_url = str(payload.get("source_url") or "")
    parsed_metadata = extract_document_metadata(content, title=title, source_url=source_url)
    metadata = {**parsed_metadata, **sanitize_metadata(dict(payload.get("metadata") or {}))}
    metadata["chunking_strategy"] = normalize_chunking_strategy(payload.get("chunking_strategy") or metadata.get("chunking_strategy"))
    for key in ["region_code", "policy_version", "effective_start_time", "effective_end_time"]:
        if payload.get(key) not in ("", None) and key not in metadata:
            metadata[key] = payload.get(key)
    if metadata.get("region_code") and not metadata.get("region_key"):
        metadata["region_key"] = metadata["region_code"]
    if metadata.get("policy_version") and not metadata.get("source_version"):
        metadata["source_version"] = metadata["policy_version"]
    document = KnowledgeDocument(
        id=str(payload["document_id"]),
        title=title,
        filename=filename,
        content_type=content_type,
        content=content,
        region_code=str(metadata.get("region_code") or metadata.get("region_key") or ""),
        policy_version=str(metadata.get("policy_version") or metadata.get("source_version") or ""),
        source_url=source_url or str(metadata.get("official_source_url") or metadata.get("source_url") or ""),
        audit_node=str(payload.get("audit_node") or ""),
        status=str(payload.get("status") or "published"),
        version_group_uid=str(payload.get("version_group_uid") or payload["document_id"]),
        version_no=int(payload.get("version_no") or 1),
        rollout_percent=max(0, min(100, int(payload.get("rollout_percent") if payload.get("rollout_percent") is not None else 100))),
        change_note=str(payload.get("change_note") or ""),
        effective_start_time=parse_optional_date(metadata.get("effective_start_time")),
        effective_end_time=parse_optional_date(metadata.get("effective_end_time")),
        metadata=metadata,
    )
    chunks_indexed = index_document_for_workflow(workflow, document)
    return {
        "document_id": document.id,
        "chunks_indexed": chunks_indexed,
        "filename": filename,
        "content_type": content_type,
        "chunking_strategy": metadata["chunking_strategy"],
        "message": "Document indexed",
    }


def parse_optional_date(value) -> date | None:
    """将可选输入解析为日期对象。"""
    if isinstance(value, date):
        return value
    if not value:
        return None
    return date.fromisoformat(str(value)[:10])

