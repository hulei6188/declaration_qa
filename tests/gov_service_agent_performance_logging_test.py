import json
import logging

import pytest

from gov_service_agent.observability.performance import performance_request, performance_stage


LOGGER_NAME = "uvicorn.error.gov_service_agent.performance"


def performance_payloads(caplog):
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == LOGGER_NAME
    ]


def test_performance_logs_include_request_correlation_and_stage_fields(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with performance_request("workflow.stream", trace_id="trace-1", session_id="session-1"):
        with performance_stage("dashscope.rerank", candidate_count=30):
            pass

    payloads = performance_payloads(caplog)
    assert [item["stage"] for item in payloads] == ["dashscope.rerank", "workflow.stream"]
    assert payloads[0]["trace_id"] == "trace-1"
    assert payloads[0]["session_id"] == "session-1"
    assert payloads[0]["candidate_count"] == 30
    assert payloads[0]["status"] == "ok"
    assert payloads[0]["elapsed_ms"] >= 0


def test_performance_stage_logs_errors_before_reraising(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with pytest.raises(RuntimeError, match="boom"):
        with performance_stage("milvus.search_batch", query_count=2):
            raise RuntimeError("boom")

    payload = performance_payloads(caplog)[0]
    assert payload["stage"] == "milvus.search_batch"
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
