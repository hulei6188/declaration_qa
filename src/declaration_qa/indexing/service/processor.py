"""索引任务分发处理。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.indexing.jobs import IndexJobStore
from declaration_qa.indexing.service.delete import process_document_delete
from declaration_qa.indexing.service.documents import process_document_index
from declaration_qa.indexing.service.reindex import process_reindex

if TYPE_CHECKING:
    from declaration_qa.application.workflow import DeclarationQAWorkflow


def process_index_job(
    job_uid: str,
    *,
    workflow: DeclarationQAWorkflow,
    job_store: IndexJobStore,
    settings: DeclarationQASettings,
) -> dict[str, Any]:
    """执行索引任务并更新任务状态。"""
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

