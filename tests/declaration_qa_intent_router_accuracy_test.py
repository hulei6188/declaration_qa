import json
import os
from pathlib import Path
from typing import Any

import pytest

from declaration_qa.question_planner import RuleQuestionPlanner
from declaration_qa.router import classify_knowledge_intent, classify_ticket_intent


DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "materials"
    / "curated"
    / "declaration_qa"
    / "2026-06-15-ahzwfw-real-40-topics"
)
DATASET_PATH = DATASET_DIR / "rag_evaluation_dataset.json"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
MIN_INTENT_ACCURACY = 0.95
MIN_CONTEXT_ACCURACY = 0.98
MIN_JOINT_ACCURACY = 0.95


def test_router_accuracy_dataset_covers_real_ahzwfw_40_topic_documents():
    manifest = load_json(MANIFEST_PATH)
    dataset = load_json(DATASET_PATH)

    manifest_docs = {
        Path(document["files"]["markdown"]).name
        for document in manifest["documents"]
        if document.get("files", {}).get("markdown")
    }
    case_docs = {
        tag.removeprefix("doc:")
        for case in dataset["cases"]
        for tag in case.get("tags") or []
        if tag.startswith("doc:")
    }

    assert len(manifest_docs) == 40
    assert len(dataset["cases"]) == 200
    assert case_docs == manifest_docs


def test_rule_question_planner_context_accuracy_on_real_ahzwfw_40_topic_dataset():
    cases = load_json(DATASET_PATH)["cases"]
    planner = RuleQuestionPlanner()

    results = [evaluate_case(planner, index, case) for index, case in enumerate(cases, start=1)]
    metrics = calculate_metrics(results)

    assert metrics["case_count"] == 200
    assert metrics["intent_accuracy"] >= MIN_INTENT_ACCURACY, format_failures(results, kind="intent")
    assert metrics["context_accuracy"] >= MIN_CONTEXT_ACCURACY, format_failures(results, kind="context")
    assert metrics["joint_accuracy"] >= MIN_JOINT_ACCURACY, format_failures(results, kind="joint")


def test_configured_question_planner_context_accuracy_on_real_ahzwfw_40_topic_dataset():
    if os.getenv("DECLARATION_QA_RUN_LIVE_ROUTER_ACCURACY") != "1":
        pytest.skip("Set DECLARATION_QA_RUN_LIVE_ROUTER_ACCURACY=1 to evaluate the configured live planner.")

    from declaration_qa.config import DeclarationQASettings
    from declaration_qa.model_providers import build_question_planner

    cases = load_json(DATASET_PATH)["cases"]
    planner = build_question_planner(DeclarationQASettings())

    results = [evaluate_case(planner, index, case) for index, case in enumerate(cases, start=1)]
    metrics = calculate_metrics(results)

    assert metrics["case_count"] == 200
    assert metrics["intent_accuracy"] >= MIN_INTENT_ACCURACY, format_failures(results, kind="intent")
    assert metrics["context_accuracy"] >= MIN_CONTEXT_ACCURACY, format_failures(results, kind="context")
    assert metrics["joint_accuracy"] >= MIN_JOINT_ACCURACY, format_failures(results, kind="joint")


def evaluate_case(planner: RuleQuestionPlanner, index: int, case: dict[str, Any]) -> dict[str, Any]:
    plan = planner.plan(router_state_from_case(case))
    first_question = (plan.questions or [{}])[0]
    expected_intent = str(case.get("expected_intent") or "")
    expected_requires_context = expected_requires_context_from_case(case)
    query = str(first_question.get("query") or case.get("query") or "")
    actual_intent = derive_intent(query, bool(first_question.get("requires_ticket_context")))
    intent_match = actual_intent == expected_intent
    context_match = bool(first_question.get("requires_ticket_context")) == expected_requires_context
    return {
        "index": index,
        "query": str(case.get("query") or ""),
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "expected_context": expected_requires_context,
        "actual_context": bool(first_question.get("requires_ticket_context")),
        "reason": plan.reason,
        "confidence": 1.0,
        "intent_match": intent_match,
        "context_match": context_match,
        "joint_match": intent_match and context_match,
    }


def router_state_from_case(case: dict[str, Any]) -> dict[str, Any]:
    query = str(case.get("query") or "")
    return {
        "query": query,
        "normalized_query": query,
        "user_id": int(case.get("user_id") or 1),
        "region_code": str(case.get("region_code") or "340000"),
        "policy_version": str(case.get("policy_version") or "2025"),
        "biz_type": str(case.get("biz_type") or ""),
        "apply_id": str(case.get("apply_id") or ""),
        "apply_context": case.get("apply_context") or {},
    }


def expected_requires_context_from_case(case: dict[str, Any]) -> bool:
    if case.get("expected_route"):
        return str(case["expected_route"]) == "ticket_context_related"
    if case.get("apply_id") or case.get("apply_context"):
        return True
    return False


def derive_intent(query: str, requires_context: bool) -> str:
    ticket_intent = classify_ticket_intent(query)
    if requires_context:
        return ticket_intent or classify_knowledge_intent(query)
    if ticket_intent == "rejection_reason":
        return "correction_guidance"
    return classify_knowledge_intent(query)


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(results),
        "intent_accuracy": ratio(item["intent_match"] for item in results),
        "context_accuracy": ratio(item["context_match"] for item in results),
        "joint_accuracy": ratio(item["joint_match"] for item in results),
    }


def ratio(values) -> float:
    values = list(values)
    return round(sum(1 for value in values if value) / len(values), 4)


def format_failures(results: list[dict[str, Any]], *, kind: str) -> str:
    key = f"{kind}_match"
    failures = [item for item in results if not item[key]]
    lines = [
        f"{kind} accuracy failed with {len(failures)} misses.",
        "First misses:",
    ]
    for item in failures[:10]:
        lines.append(
            "#%(index)s expected_context=%(expected_context)s actual_context=%(actual_context)s "
            "expected_intent=%(expected_intent)s actual_intent=%(actual_intent)s "
            "confidence=%(confidence)s reason=%(reason)s query=%(query)s" % item
        )
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
