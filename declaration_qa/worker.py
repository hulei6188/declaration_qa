"""RQ 索引 worker 启动入口。"""

from __future__ import annotations

import os
from typing import Type

from declaration_qa.config import get_settings


def select_rq_worker_class(worker_class: Type, simple_worker_class: Type, timer_death_penalty_class: Type) -> Type:
    """选择可用的 RQ Worker 类。"""
    if hasattr(os, "fork"):
        return worker_class
    simple_worker_class.death_penalty_class = timer_death_penalty_class
    return simple_worker_class


def main() -> None:
    """启动索引任务 worker。"""
    settings = get_settings()
    if settings.index_queue_backend.strip().lower() != "rq":
        raise SystemExit("DECLARATION_QA_INDEX_QUEUE_BACKEND=rq is required to run the worker")
    if not settings.effective_redis_url:
        raise SystemExit("DECLARATION_QA_REDIS_URL is required to run the worker")

    from redis import Redis
    from rq import Queue, SimpleWorker, Worker
    from rq.timeouts import TimerDeathPenalty

    redis = Redis.from_url(settings.effective_redis_url)
    queue = Queue(settings.index_queue_name, connection=redis)
    worker_class = select_rq_worker_class(Worker, SimpleWorker, TimerDeathPenalty)
    worker = worker_class([queue], connection=redis)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
