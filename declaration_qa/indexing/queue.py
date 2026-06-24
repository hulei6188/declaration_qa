"""知识库索引任务队列抽象和实现。"""

from __future__ import annotations

from typing import Callable, Protocol

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.indexing.jobs import IndexJobStore


class IndexQueue(Protocol):
    """索引任务投递队列协议。"""
    backend: str

    def enqueue(self, job_uid: str) -> dict:
        """投递索引任务并返回队列信息。"""
        ...


class InlineIndexQueue:
    """同步执行索引任务的队列实现。"""
    backend = "inline"

    def __init__(self, handler: Callable[[str], dict]) -> None:
        """初始化InlineIndexQueue。"""
        self._handler = handler

    def enqueue(self, job_uid: str) -> dict:
        """投递索引任务并返回队列信息。"""
        result = self._handler(job_uid)
        return {"backend": self.backend, "job_uid": job_uid, "result": result}


class RQIndexQueue:
    """基于 RQ 和 Redis 的异步索引队列实现。"""
    backend = "rq"

    def __init__(self, settings: DeclarationQASettings) -> None:
        """初始化RQIndexQueue。"""
        if not settings.effective_redis_url:
            raise ValueError("DECLARATION_QA_REDIS_URL is required for RQ index queue")
        from redis import Redis
        from rq import Queue

        self._queue = Queue(
            settings.index_queue_name,
            connection=Redis.from_url(settings.effective_redis_url),
            default_timeout=settings.index_job_timeout_seconds,
        )
        self._job_timeout = settings.index_job_timeout_seconds

    def enqueue(self, job_uid: str) -> dict:
        """投递索引任务并返回队列信息。"""
        rq_job = self._queue.enqueue(
            "declaration_qa.indexing.tasks.process_index_job_task",
            job_uid,
            job_timeout=self._job_timeout,
            result_ttl=86400,
            failure_ttl=604800,
        )
        return {"backend": self.backend, "job_uid": job_uid, "rq_job_id": rq_job.id}


def build_index_queue(
    settings: DeclarationQASettings,
    *,
    inline_handler: Callable[[str], dict],
    job_store: IndexJobStore,
) -> IndexQueue:
    """按配置创建索引任务队列。"""
    backend = settings.index_queue_backend.strip().lower()
    if backend == "rq":
        return RQIndexQueue(settings)
    if backend == "inline":
        return InlineIndexQueue(inline_handler)
    raise ValueError(f"Unsupported index queue backend: {settings.index_queue_backend}")
