"""独立RQ评测Worker入口。"""

from __future__ import annotations

from gov_service_agent.core.config import get_settings
from gov_service_agent.indexing.worker import select_rq_worker_class


def main() -> None:
    settings = get_settings()
    if settings.evaluation_queue_backend.strip().lower() != "rq":
        raise SystemExit("GOV_SERVICE_AGENT_EVAL_QUEUE_BACKEND=rq is required")
    if not settings.effective_redis_url:
        raise SystemExit("GOV_SERVICE_AGENT_REDIS_URL is required")
    from redis import Redis
    from rq import Queue, SimpleWorker, Worker
    from rq.timeouts import TimerDeathPenalty

    redis = Redis.from_url(settings.effective_redis_url)
    queue = Queue(settings.evaluation_queue_name, connection=redis)
    worker_class = select_rq_worker_class(Worker, SimpleWorker, TimerDeathPenalty)
    worker_class([queue], connection=redis).work(with_scheduler=True)


if __name__ == "__main__":
    main()
