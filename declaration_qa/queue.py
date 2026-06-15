from __future__ import annotations

from typing import Callable, Protocol

from declaration_qa.config import DeclarationQASettings
from declaration_qa.indexing import IndexJobStore


class IndexQueue(Protocol):
    backend: str

    def enqueue(self, job_uid: str) -> dict:
        ...


class InlineIndexQueue:
    backend = "inline"

    def __init__(self, handler: Callable[[str], dict]) -> None:
        self._handler = handler

    def enqueue(self, job_uid: str) -> dict:
        result = self._handler(job_uid)
        return {"backend": self.backend, "job_uid": job_uid, "result": result}


class RQIndexQueue:
    backend = "rq"

    def __init__(self, settings: DeclarationQASettings) -> None:
        if not settings.effective_redis_url:
            raise ValueError("DECLARATION_QA_REDIS_URL is required for RQ index queue")
        from redis import Redis
        from rq import Queue

        self._queue = Queue(
            settings.index_queue_name,
            connection=Redis.from_url(settings.effective_redis_url),
            default_timeout=settings.index_job_timeout_seconds,
        )

    def enqueue(self, job_uid: str) -> dict:
        rq_job = self._queue.enqueue(
            "declaration_qa.worker_tasks.process_index_job_task",
            job_uid,
            job_timeout=self._queue.default_timeout,
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
    backend = settings.index_queue_backend.strip().lower()
    if backend == "rq":
        return RQIndexQueue(settings)
    if backend == "inline":
        return InlineIndexQueue(inline_handler)
    raise ValueError(f"Unsupported index queue backend: {settings.index_queue_backend}")
