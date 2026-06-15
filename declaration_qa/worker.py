from __future__ import annotations

from declaration_qa.config import get_settings


def main() -> None:
    settings = get_settings()
    if settings.index_queue_backend.strip().lower() != "rq":
        raise SystemExit("DECLARATION_QA_INDEX_QUEUE_BACKEND=rq is required to run the worker")
    if not settings.effective_redis_url:
        raise SystemExit("DECLARATION_QA_REDIS_URL is required to run the worker")

    from redis import Redis
    from rq import Queue, Worker

    redis = Redis.from_url(settings.effective_redis_url)
    queue = Queue(settings.index_queue_name, connection=redis)
    worker = Worker([queue], connection=redis)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
