import json
from pathlib import Path

from declaration_qa.api.schemas import EvaluationCaseRequest
from declaration_qa.application.evals.comparison import compare_runs
from declaration_qa.application.evals.evaluators import evaluate_answer, evaluate_retrieval
from declaration_qa.application.evals.gate import evaluate_gate
from declaration_qa.application.evals.fixtures import EvaluationFixtureManager
from declaration_qa.application.evals.pii import sanitize_text
from declaration_qa.application.evals.runner import EvaluationRunner
from declaration_qa.application.evals.trajectory import evaluate_trajectory
from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.core.runtime import production_readiness_errors


ROOT = Path(__file__).resolve().parents[1]


def tool_call(name, arguments, result, *, sequence, ok=True):
    return {
        "turn_index": 1,
        "sequence_no": sequence,
        "tool_name": name,
        "arguments": arguments,
        "status": "success",
        "ok": ok,
        "result": result,
    }


def test_agent_trajectory_accepts_dependencies_provenance_and_human_confirmation():
    actual = {
        "calls": [
            tool_call("knowledge_search", {}, {"data": {"sources": [{"item_code": "ITEM-1"}]}}, sequence=1),
            tool_call("appointment_hall_query", {"item_code": "ITEM-1"}, {"data": {"halls": [{"hall_code": "H-1"}]}}, sequence=2),
            tool_call("appointment_slot_query", {"item_code": "ITEM-1", "hall_codes": ["H-1"]}, {"data": {"slots": [{"slot_id": 41}]}}, sequence=3),
            tool_call("appointment_book", {"slot_id": 41}, {"data": {"human_confirmation": {"completed": True}}}, sequence=4),
        ]
    }
    expected = {
        "required_tools": ["knowledge_search", "appointment_hall_query", "appointment_slot_query", "appointment_book"],
        "dependencies": {
            "appointment_hall_query": ["knowledge_search"],
            "appointment_slot_query": ["appointment_hall_query"],
            "appointment_book": ["appointment_slot_query"],
        },
        "requires_confirmation": ["appointment_book"],
    }

    scores, failures, _ = evaluate_trajectory(actual, expected, user_id=900001)

    assert failures == []
    assert scores["tool_dependency_pass"] == 1.0
    assert scores["argument_provenance_pass"] == 1.0
    assert scores["human_confirmation_pass"] == 1.0


def test_agent_trajectory_blocks_fabricated_slot_and_unconfirmed_write():
    actual = {
        "calls": [
            tool_call("appointment_slot_query", {"item_code": "ITEM-1", "hall_codes": ["H-1"]}, {"data": {"slots": [{"slot_id": 41}]}}, sequence=1),
            tool_call("appointment_book", {"slot_id": 999}, {"data": {"appointment": {"appointment_id": 1}}}, sequence=2),
        ]
    }
    scores, failures, _ = evaluate_trajectory(
        actual,
        {"requires_confirmation": ["appointment_book"]},
        user_id=900001,
    )

    assert "untrusted_argument:appointment_book.slot_id" in failures
    assert "write_without_confirmation:appointment_book" in failures
    assert scores["argument_provenance_pass"] == 0.0
    assert evaluate_gate({}, failure_reasons=failures)["passed"] is False


def test_retrieval_metrics_distinguish_hit_recall_mrr_and_duplicates():
    sources = [
        {"doc_id": "doc-x", "chunk_id": "chunk-x"},
        {"doc_id": "doc-a", "chunk_id": "chunk-a"},
        {"doc_id": "doc-a", "chunk_id": "chunk-a"},
    ]
    scores, failures, _ = evaluate_retrieval(
        sources,
        expected_doc_ids=[],
        expected_chunk_ids=["chunk-a", "chunk-b"],
        top_k=3,
    )

    assert failures == []
    assert scores["retrieval_hit_at_k"] == 1.0
    assert scores["retrieval_recall_at_k"] == 0.5
    assert scores["retrieval_mrr"] == 0.5
    assert scores["retrieval_duplicate_rate"] > 0


def test_not_applicable_answer_scores_do_not_raise_or_count_as_failures():
    scores, failures, records = evaluate_answer(
        answer="没有可引用的证据。",
        refused=False,
        expected_refused=None,
        expected_keywords=[],
        sources=[],
    )

    assert scores["answer_keyword_coverage"] is None
    assert failures == []
    assert all(record["label"] == "not_applicable" for record in records)
    assert all(record["score"] is None for record in records)


def test_fixture_manager_does_not_touch_cases_without_fixture_key():
    class ExplodingEngine:
        def begin(self):
            raise AssertionError("fixture manager must not open the database")

    manager = EvaluationFixtureManager(type("Store", (), {"engine": ExplodingEngine()})())

    manager.prepare({"user_id": 1, "fixture_key": ""})
    manager.cleanup({"user_id": 1, "fixture_key": ""})


def test_evaluation_scores_are_written_to_each_langfuse_trace(monkeypatch):
    captured = []

    def capture(trace_id, **payload):
        captured.append((trace_id, payload))

    monkeypatch.setattr("declaration_qa.application.evals.runner.score_trace", capture)
    runner = EvaluationRunner(
        workflow=type("Workflow", (), {"audit_store": object()})(),
        store=object(),
        langfuse_settings=object(),
    )
    runner._score_langfuse_traces(
        {"case_id": "case-1", "severity": "critical"},
        {"passed": True, "actual_trajectory": {"trace_ids": ["trace-1", "trace-2"]}},
        [{"metric_name": "tool_dependency_pass", "score": 1.0, "reason": ""}],
    )

    assert [(trace_id, payload["name"]) for trace_id, payload in captured] == [
        ("trace-1", "eval.tool_dependency_pass"),
        ("trace-1", "eval.case_passed"),
        ("trace-2", "eval.tool_dependency_pass"),
        ("trace-2", "eval.case_passed"),
    ]


def test_candidate_text_is_sanitized_before_persistence():
    value = sanitize_text("身份证340101199001011234，手机13800138000，邮箱a@example.com", salt="test")
    assert "340101199001011234" not in value
    assert "13800138000" not in value
    assert "a@example.com" not in value
    assert "[ID_CARD]" in value


def test_run_comparison_marks_quality_drop_as_regression_and_latency_drop_as_improvement():
    result = compare_runs(
        {"run_id": "base", "metrics": {"task_success_rate": 0.98, "performance": {"p95_latency_ms": 2000}}},
        {"run_id": "candidate", "metrics": {"task_success_rate": 0.90, "performance": {"p95_latency_ms": 1500}}},
    )
    regression_names = {item["metric"] for item in result["regressions"]}
    improvement_names = {item["metric"] for item in result["improvements"]}
    assert "task_success_rate" in regression_names
    assert "performance.p95_latency_ms" in improvement_names


def test_agent_golden_v1_contains_valid_20_15_15_10_case_split():
    payload = json.loads((ROOT / "materials/evaluation/agent-golden-v1.json").read_text(encoding="utf-8"))
    counts = {}
    for case in payload["cases"]:
        EvaluationCaseRequest.model_validate(case)
        counts[case["tags"][0]] = counts.get(case["tags"][0], 0) + 1
    assert len(payload["cases"]) == 60
    assert counts == {"appointment": 20, "business": 15, "security": 15, "multi_turn": 10}


def test_production_requires_isolated_eval_database_queue_and_pii_salt():
    settings = DeclarationQASettings(
        DECLARATION_QA_ENV="production",
        DECLARATION_QA_DATABASE_URL="postgresql+psycopg://prod",
        DECLARATION_QA_EVAL_DATABASE_URL="postgresql+psycopg://prod",
        DECLARATION_QA_EVAL_QUEUE_BACKEND="inline",
    )
    errors = production_readiness_errors(settings)
    assert "DECLARATION_QA_EVAL_DATABASE_URL must differ from DECLARATION_QA_DATABASE_URL in production" in errors
    assert "DECLARATION_QA_EVAL_QUEUE_BACKEND=rq is required when evaluation is enabled in production" in errors
    assert "DECLARATION_QA_EVAL_PII_SALT is required when evaluation is enabled in production" in errors
