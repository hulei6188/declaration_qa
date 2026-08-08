import json
from pathlib import Path

from gov_service_agent.api.schemas import EvaluationCaseRequest
from gov_service_agent.application.evals.comparison import compare_runs
from gov_service_agent.application.evals.candidates import infer_attribution_category
from gov_service_agent.application.evals.evaluators import evaluate_answer, evaluate_retrieval
from gov_service_agent.application.evals.gate import evaluate_gate
from gov_service_agent.application.evals.fixtures import EvaluationFixtureManager
from gov_service_agent.application.evals.pii import sanitize_text
from gov_service_agent.application.evals.runner import EvaluationRunner
from gov_service_agent.application.evals.trajectory import evaluate_trajectory
from gov_service_agent.application.helpers.actions import suggested_actions
from gov_service_agent.core.config import GovServiceAgentSettings
from gov_service_agent.core.runtime import production_readiness_errors
from gov_service_agent.storage.audit import InMemoryAuditStore


ROOT = Path(__file__).resolve().parents[1]


def test_badcase_attribution_and_knowledge_draft_lifecycle():
    category, _ = infer_attribution_category({"signals": {"tool_error": True}, "sources": []})
    assert category == "tool_error"
    category, _ = infer_attribution_category({"signals": {}, "sources": []})
    assert category == "retrieval_miss"

    store = InMemoryAuditStore()
    candidate = store.create_eval_candidate({
        "audit_uid": "audit-1", "source": "feedback", "severity": "medium",
        "sanitized_query": "如何办理", "sanitized_answer": "未找到",
        "attribution_category": "knowledge_missing",
    })
    candidate_id = candidate["candidate_id"]
    store.review_eval_candidate(
        candidate_id=candidate_id, status="labeled", severity="medium",
        labels=["knowledge_missing"], comment="人工归因", reviewer_id=1,
        attribution_category="knowledge_missing",
    )
    store.create_knowledge_draft(candidate_id=candidate_id, draft={
        "category": "knowledge_missing", "title": "办理 FAQ", "content": "办理需要身份证。",
    })
    approved = store.review_knowledge_draft(candidate_id=candidate_id, status="approved", comment="核实通过", reviewer_id=1)
    assert approved["knowledge_draft"]["status"] == "approved"
    published = store.publish_knowledge_draft(candidate_id=candidate_id, document_id="doc-faq-1")
    assert published["knowledge_draft"]["status"] == "published"
    assert published["knowledge_draft"]["knowledge_document_id"] == "doc-faq-1"


def test_human_service_ticket_carries_agent_context_and_supports_claim_reply():
    store = InMemoryAuditStore()
    audit_id = store.append({
        "user_id": 7, "session_id": "session-1", "query": "进度查不到",
        "normalized_query": "查询办件进度", "intent": "ticket_status",
        "route": "business", "business_snapshot": {"application_id": 42},
        "dialog_task": {"status": "collecting", "values": {"name": "张三"}},
        "retrieved_sources": [{"doc_id": "doc-1", "snippet": "办理进度说明"}],
        "tool_results": [{"tool_name": "ticket_detail_query", "status": "not_found", "ok": False}],
        "refused": True,
    })
    ticket = store.create_human_service_ticket(audit_id=audit_id, user_id=7, session_id="session-1", note="请人工核验")
    assert ticket["status"] == "open"
    assert ticket["context"]["business_snapshot"]["application_id"] == 42
    assert ticket["context"]["dialog_task"]["values"]["name"] == "张三"
    ticket = store.claim_human_service_ticket(ticket_id=ticket["ticket_id"], agent_id=99)
    assert ticket["status"] == "claimed"
    ticket = store.add_human_service_message(ticket_id=ticket["ticket_id"], role="agent", content="我来帮您核验", author_id=99)
    assert ticket["messages"][-1]["content"] == "我来帮您核验"


def test_explicit_user_handoff_request_exposes_human_service_action():
    actions = suggested_actions({"query": "请帮我转人工客服", "app_version": "v1"})
    assert actions[0]["type"] == "human_service"
    assert actions[0]["payload"]["reason"] == "user_requested"


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

    monkeypatch.setattr("gov_service_agent.application.evals.runner.score_trace", capture)
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
    settings = GovServiceAgentSettings(
        GOV_SERVICE_AGENT_ENV="production",
        GOV_SERVICE_AGENT_DATABASE_URL="postgresql+psycopg://prod",
        GOV_SERVICE_AGENT_EVAL_DATABASE_URL="postgresql+psycopg://prod",
        GOV_SERVICE_AGENT_EVAL_QUEUE_BACKEND="inline",
    )
    errors = production_readiness_errors(settings)
    assert "GOV_SERVICE_AGENT_EVAL_DATABASE_URL must differ from GOV_SERVICE_AGENT_DATABASE_URL in production" in errors
    assert "GOV_SERVICE_AGENT_EVAL_QUEUE_BACKEND=rq is required when evaluation is enabled in production" in errors
    assert "GOV_SERVICE_AGENT_EVAL_PII_SALT is required when evaluation is enabled in production" in errors
