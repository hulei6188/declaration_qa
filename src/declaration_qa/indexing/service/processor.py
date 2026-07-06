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
    stage_history = list((payload.get("etl") or {}).get("stages") or [])

    def update_stage(stage: str, progress: int, message: str, details: dict[str, Any] | None = None) -> None:
        nonlocal payload, stage_history
        event = {
            "stage": stage,
            "progress": max(0, min(100, int(progress))),
            "message": message,
            "details": details or {},
        }
        stage_history = [*stage_history, event]
        payload = {
            **payload,
            "etl": {
                "current_stage": stage,
                "stages": stage_history,
                "last_message": message,
                "last_details": details or {},
            },
        }
        job_store.update_index_job(
            job_uid=job_uid,
            status="running",
            progress=progress,
            message=message,
            payload=payload,
        )

    update_stage("accepted", 10, "Index worker accepted ETL task", {"job_type": job_type})
    try:
        if job_type == "document_index":
            result = process_document_index(
                payload,
                workflow=workflow,
                settings=settings,
                job_uid=job_uid,
                stage_callback=update_stage,
            )
        elif job_type == "knowledge_base_reindex":
            update_stage("reindex", 40, "Knowledge base reindex started", {})
            result = process_reindex(workflow)
        elif job_type == "document_delete":
            update_stage("delete", 40, "Document delete started", {"document_id": payload.get("document_id") or ""})
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
        payload = {
            **payload,
            "etl": {
                **(payload.get("etl") or {}),
                "current_stage": "failed",
                "last_message": str(exc),
            },
        }
        return job_store.update_index_job(
            job_uid=job_uid,
            status="failed",
            progress=100,
            error_message=str(exc),
            payload=payload,
        )
