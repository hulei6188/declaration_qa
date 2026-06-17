"""后台索引任务执行入口。"""

from __future__ import annotations

from declaration_qa.config import get_settings
from declaration_qa.factory import create_index_job_store, create_workflow
from declaration_qa.index_service import process_index_job


def process_index_job_task(job_uid: str) -> dict:
    """创建工作流并处理后台索引任务。"""
    settings = get_settings()
    workflow = create_workflow(settings)
    job_store = create_index_job_store(workflow)
    return process_index_job(job_uid, workflow=workflow, job_store=job_store, settings=settings)
