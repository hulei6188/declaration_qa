"""评测运行的内联与RQ队列。"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class EvaluationQueue(Protocol):
    backend: str

    def enqueue(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class InlineEvaluationQueue:
    backend = "inline"

    def __init__(self, handler: Callable[[str, dict[str, Any]], dict[str, Any]]) -> None:
        self.handler = handler

    def enqueue(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"backend": self.backend, "run_id": run_id, "result": self.handler(run_id, payload)}


class RQEvaluationQueue:
    backend = "rq"

    def __init__(self, settings) -> None:
        if not settings.effective_redis_url:
            raise ValueError("GOV_SERVICE_AGENT_REDIS_URL is required for RQ evaluation queue")
        from redis import Redis
        from rq import Queue

        self.queue = Queue(
            settings.evaluation_queue_name,
            connection=Redis.from_url(settings.effective_redis_url),
            default_timeout=settings.evaluation_job_timeout_seconds,
        )
        self.timeout = settings.evaluation_job_timeout_seconds

    def enqueue(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = self.queue.enqueue(
            "gov_service_agent.application.evals.tasks.process_evaluation_run_task",
            run_id,
            payload,
            job_timeout=self.timeout,
            result_ttl=86400,
            failure_ttl=604800,
        )
        return {"backend": self.backend, "run_id": run_id, "rq_job_id": job.id}


def build_evaluation_queue(settings, *, inline_handler) -> EvaluationQueue:
    backend = settings.evaluation_queue_backend.strip().lower()
    if backend == "rq":
        return RQEvaluationQueue(settings)
    if backend == "inline":
        return InlineEvaluationQueue(inline_handler)
    raise ValueError(f"Unsupported evaluation queue backend: {settings.evaluation_queue_backend}")
