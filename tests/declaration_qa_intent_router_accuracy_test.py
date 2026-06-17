import json
import os
from pathlib import Path
from typing import Any

import pytest

from declaration_qa.router import (
    ROUTE_TICKET_CONTEXT_RELATED,
    ROUTE_TICKET_CONTEXT_UNRELATED,
    IntentRouter,
)


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
MIN_ROUTE_ACCURACY = 0.98
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


def test_rule_intent_router_accuracy_on_real_ahzwfw_40_topic_dataset():
    cases = load_json(DATASET_PATH)["cases"]
    router = IntentRouter()

    results = [evaluate_case(router, index, case) for index, case in enumerate(cases, start=1)]
    metrics = calculate_metrics(results)

    assert metrics["case_count"] == 200
    assert metrics["intent_accuracy"] >= MIN_INTENT_ACCURACY, format_failures(results, kind="intent")
    assert metrics["route_accuracy"] >= MIN_ROUTE_ACCURACY, format_failures(results, kind="route")
    assert metrics["joint_accuracy"] >= MIN_JOINT_ACCURACY, format_failures(results, kind="joint")


def test_configured_intent_router_accuracy_on_real_ahzwfw_40_topic_dataset():
    if os.getenv("DECLARATION_QA_RUN_LIVE_ROUTER_ACCURACY") != "1":
        pytest.skip("Set DECLARATION_QA_RUN_LIVE_ROUTER_ACCURACY=1 to evaluate the configured live router.")

    from declaration_qa.config import DeclarationQASettings
    from declaration_qa.model_providers import build_intent_router

    cases = load_json(DATASET_PATH)["cases"]
    router = build_intent_router(DeclarationQASettings())

    results = [evaluate_case(router, index, case) for index, case in enumerate(cases, start=1)]
    metrics = calculate_metrics(results)

    assert metrics["case_count"] == 200
    assert metrics["intent_accuracy"] >= MIN_INTENT_ACCURACY, format_failures(results, kind="intent")
    assert metrics["route_accuracy"] >= MIN_ROUTE_ACCURACY, format_failures(results, kind="route")
    assert metrics["joint_accuracy"] >= MIN_JOINT_ACCURACY, format_failures(results, kind="joint")


def evaluate_case(router: IntentRouter, index: int, case: dict[str, Any]) -> dict[str, Any]:
    decision = router.route(router_state_from_case(case))
    expected_intent = str(case.get("expected_intent") or "")
    expected_route = expected_route_from_case(case)
    intent_match = decision.intent == expected_intent
    route_match = decision.route == expected_route
    return {
        "index": index,
        "query": str(case.get("query") or ""),
        "expected_intent": expected_intent,
        "actual_intent": decision.intent,
        "expected_route": expected_route,
        "actual_route": decision.route,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "intent_match": intent_match,
        "route_match": route_match,
        "joint_match": intent_match and route_match,
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


def expected_route_from_case(case: dict[str, Any]) -> str:
    if case.get("expected_route"):
        return str(case["expected_route"])
    if case.get("apply_id") or case.get("apply_context"):
        return ROUTE_TICKET_CONTEXT_RELATED
    return ROUTE_TICKET_CONTEXT_UNRELATED


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(results),
        "intent_accuracy": ratio(item["intent_match"] for item in results),
        "route_accuracy": ratio(item["route_match"] for item in results),
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
            "#%(index)s expected_route=%(expected_route)s actual_route=%(actual_route)s "
            "expected_intent=%(expected_intent)s actual_intent=%(actual_intent)s "
            "confidence=%(confidence)s reason=%(reason)s query=%(query)s" % item
        )
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
