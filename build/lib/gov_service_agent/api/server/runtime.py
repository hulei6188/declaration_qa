"""服务运行时组件。"""

from __future__ import annotations

from gov_service_agent.application.factory import create_index_job_store, create_workflow
from gov_service_agent.core.config import get_settings
from gov_service_agent.core.runtime import assert_production_ready
from gov_service_agent.indexing.queue import build_index_queue
from gov_service_agent.indexing.service.processor import process_index_job


settings = get_settings()
assert_production_ready(settings)
workflow = create_workflow(settings)
index_job_store = create_index_job_store(workflow)


def process_index_job_inline(job_uid: str) -> dict:
    """以内联方式处理索引任务。"""
    return process_index_job(job_uid, workflow=workflow, job_store=index_job_store, settings=settings)


index_queue = build_index_queue(settings, inline_handler=process_index_job_inline, job_store=index_job_store)
