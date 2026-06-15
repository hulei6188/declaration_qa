from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from declaration_qa.config import DeclarationQASettings
from declaration_qa.documents import HttpOcrClient, parse_document_bytes
from declaration_qa.domain import KnowledgeDocument
from declaration_qa.indexing import IndexJobStore
from declaration_qa.workflow import DeclarationQAWorkflow


def index_document_for_workflow(workflow: DeclarationQAWorkflow, document: KnowledgeDocument) -> int:
    chunks_indexed = workflow.rag_engine.index_document(document)
    if hasattr(workflow.audit_store, "persist_document"):
        workflow.audit_store.persist_document(document, workflow.rag_engine.document_chunks(document))
    return chunks_indexed


def process_index_job(
    job_uid: str,
    *,
    workflow: DeclarationQAWorkflow,
    job_store: IndexJobStore,
    settings: DeclarationQASettings,
) -> dict[str, Any]:
    item = job_store.get_index_job(job_uid)
    if not item:
        raise ValueError(f"Index job not found: {job_uid}")
    job_type = str(item.get("job_type") or "")
    payload = dict(item.get("payload") or {})
    job_store.update_index_job(job_uid=job_uid, status="running", progress=10, message="Index worker started")
    try:
        if job_type == "document_index":
            result = process_document_index(payload, workflow=workflow, settings=settings)
        elif job_type == "knowledge_base_reindex":
            result = process_reindex(workflow)
        elif job_type == "document_delete":
            result = process_document_delete(payload, workflow=workflow)
        else:
            raise ValueError(f"Unsupported index job type: {job_type}")
        return job_store.update_index_job(
            job_uid=job_uid,
            status="succeeded",
            progress=100,
            message=result.get("message", "Index job succeeded"),
            payload={**payload, **result},
        )
    except Exception as exc:
        return job_store.update_index_job(
            job_uid=job_uid,
            status="failed",
            progress=100,
            error_message=str(exc),
            payload=payload,
        )


def process_document_index(
    payload: dict[str, Any],
    *,
    workflow: DeclarationQAWorkflow,
    settings: DeclarationQASettings,
) -> dict[str, Any]:
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
    document = KnowledgeDocument(
        id=str(payload["document_id"]),
        title=str(payload.get("title") or filename or payload["document_id"]),
        filename=filename,
        content_type=content_type,
        content=content,
        biz_type=str(payload.get("biz_type") or ""),
        region_code=str(payload.get("region_code") or "340000"),
        policy_version=str(payload.get("policy_version") or "2025"),
        source_type=str(payload.get("source_type") or "guide"),
        source_url=str(payload.get("source_url") or ""),
        material_code=str(payload.get("material_code") or ""),
        reject_code=str(payload.get("reject_code") or ""),
        audit_node=str(payload.get("audit_node") or ""),
        status=str(payload.get("status") or "published"),
        version_group_uid=str(payload.get("version_group_uid") or payload["document_id"]),
        version_no=int(payload.get("version_no") or 1),
        rollout_percent=max(0, min(100, int(payload.get("rollout_percent") if payload.get("rollout_percent") is not None else 100))),
        change_note=str(payload.get("change_note") or ""),
        effective_start_time=parse_optional_date(payload.get("effective_start_time")),
        effective_end_time=parse_optional_date(payload.get("effective_end_time")),
    )
    chunks_indexed = index_document_for_workflow(workflow, document)
    return {
        "document_id": document.id,
        "chunks_indexed": chunks_indexed,
        "filename": filename,
        "content_type": content_type,
        "message": "Document indexed",
    }


def process_reindex(workflow: DeclarationQAWorkflow) -> dict[str, Any]:
    documents = (
        workflow.audit_store.list_documents()
        if hasattr(workflow.audit_store, "list_documents")
        else workflow.rag_engine.documents()
    )
    chunks_indexed = workflow.rag_engine.rebuild(documents)
    return {
        "documents_indexed": len(documents),
        "chunks_indexed": chunks_indexed,
        "message": "Knowledge base reindexed",
    }


def process_document_delete(payload: dict[str, Any], *, workflow: DeclarationQAWorkflow) -> dict[str, Any]:
    document_id = str(payload.get("document_id") or "")
    if not document_id:
        raise ValueError("document_id is required")
    workflow.rag_engine.delete_document(document_id)
    deleted = workflow.audit_store.delete_document(document_id) if hasattr(workflow.audit_store, "delete_document") else True
    return {"document_id": document_id, "deleted": deleted, "message": "Document deleted"}


def parse_optional_date(value) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    return date.fromisoformat(str(value))
