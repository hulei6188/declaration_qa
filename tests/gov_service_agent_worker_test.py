from gov_service_agent.indexing.worker import select_rq_worker_class


class _ForkWorker:
    pass


class _SimpleWorker:
    pass


class _TimerDeathPenalty:
    pass


def test_rq_worker_uses_simple_worker_when_fork_is_unavailable(monkeypatch):
    monkeypatch.delattr("gov_service_agent.indexing.worker.os.fork", raising=False)

    assert select_rq_worker_class(_ForkWorker, _SimpleWorker, _TimerDeathPenalty) is _SimpleWorker
    assert _SimpleWorker.death_penalty_class is _TimerDeathPenalty


def test_rq_worker_uses_default_worker_when_fork_is_available(monkeypatch):
    monkeypatch.setattr("gov_service_agent.indexing.worker.os.fork", object(), raising=False)

    assert select_rq_worker_class(_ForkWorker, _SimpleWorker, _TimerDeathPenalty) is _ForkWorker
