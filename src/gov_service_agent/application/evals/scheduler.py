"""独立的APScheduler线上评测进程入口。"""

from __future__ import annotations

from gov_service_agent.application.evals.online_monitor import run_online_evaluation_cycle
from gov_service_agent.core.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.online_evaluation_enabled:
        raise SystemExit("GOV_SERVICE_AGENT_ONLINE_EVAL_ENABLED=true is required")
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(
        run_online_evaluation_cycle,
        trigger="interval",
        minutes=settings.online_evaluation_interval_minutes,
        kwargs={"settings": settings},
        id="gov-service-agent-online-evaluation",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=max(60, settings.online_evaluation_interval_minutes * 60),
    )
    if settings.online_evaluation_run_on_start:
        run_online_evaluation_cycle(settings)
    scheduler.start()


if __name__ == "__main__":
    main()
