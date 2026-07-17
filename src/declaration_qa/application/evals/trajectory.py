"""Agent 运行轨迹标准化和确定性约束检查。"""

from __future__ import annotations

from typing import Any


ARGUMENT_PROVENANCE = {
    "appointment_hall_query": {"item_code": [("knowledge_search", "item_code")]},
    "appointment_slot_query": {
        "item_code": [("knowledge_search", "item_code")],
        "hall_codes": [("appointment_hall_query", "hall_code")],
    },
    "appointment_book": {"slot_id": [("appointment_slot_query", "slot_id")]},
    "appointment_cancel": {"appointment_id": [("appointment_list_query", "appointment_id")]},
    "queue_status_query": {"appointment_id": [("appointment_list_query", "appointment_id")]},
    "ticket_detail_query": {"item_codes": [("ticket_list_query", "item_code")]},
    "payment_order_query": {"application_ids": [("ticket_detail_query", "application_id")]},
    "submission_receipt_query": {"application_ids": [("ticket_detail_query", "application_id")]},
    "result_certificate_query": {"application_ids": [("ticket_detail_query", "application_id")]},
    "delivery_status_query": {"application_ids": [("ticket_detail_query", "application_id")]},
}


def normalize_agent_run(run: dict[str, Any] | None, *, turn_index: int = 0) -> dict[str, Any]:
    """提取稳定、可比较的工具轨迹和运行资源统计。"""
    run = run or {}
    calls = []
    for call in run.get("tool_calls") or []:
        calls.append(
            {
                "turn_index": turn_index,
                "sequence_no": int(call.get("sequence_no") or 0),
                "tool_name": str(call.get("tool_name") or ""),
                "arguments": dict(call.get("arguments") or {}),
                "status": str(call.get("status") or ""),
                "ok": bool(call.get("ok")),
                "result": dict(call.get("result") or {}),
                "error": str(call.get("error") or ""),
                "latency_ms": call.get("latency_ms"),
            }
        )
    messages = run.get("messages") or []
    return {
        "run_id": str(run.get("run_id") or ""),
        "status": str(run.get("status") or ""),
        "trace_id": str(run.get("trace_id") or ""),
        "iteration_count": int(run.get("iteration_count") or 0),
        "prompt_version": str(run.get("prompt_version") or ""),
        "tool_registry_version": str(run.get("tool_registry_version") or ""),
        "model_snapshot": dict(run.get("model_snapshot") or {}),
        "calls": calls,
        "input_tokens": sum(int(message.get("input_tokens") or 0) for message in messages),
        "output_tokens": sum(int(message.get("output_tokens") or 0) for message in messages),
        "total_tokens": sum(int(message.get("total_tokens") or 0) for message in messages),
        "latency_ms": sum(int(call.get("latency_ms") or 0) for call in calls),
    }


def merge_turn_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": trajectories,
        "trace_ids": [item.get("trace_id") for item in trajectories if item.get("trace_id")],
        "calls": [call for trajectory in trajectories for call in trajectory.get("calls") or []],
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in trajectories),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in trajectories),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in trajectories),
        "latency_ms": sum(int(item.get("latency_ms") or 0) for item in trajectories),
    }


def evaluate_trajectory(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    user_id: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    calls = list(actual.get("calls") or [])
    names = [str(call.get("tool_name") or "") for call in calls]
    required = [str(name) for name in expected.get("required_tools") or []]
    forbidden = [str(name) for name in expected.get("forbidden_tools") or []]
    dependencies = dict(expected.get("dependencies") or expected.get("tool_dependencies") or {})
    confirmations = [str(name) for name in expected.get("requires_confirmation") or []]
    failures: list[str] = []

    missing = [name for name in required if name not in names]
    present_forbidden = [name for name in forbidden if name in names]
    if missing:
        failures.append("missing_required_tools:" + ",".join(missing))
    if present_forbidden:
        failures.append("forbidden_tools_called:" + ",".join(present_forbidden))
    max_tool_calls = expected.get("max_tool_calls")
    if max_tool_calls is not None and len(calls) > int(max_tool_calls):
        failures.append(f"tool_call_limit_exceeded:{len(calls)}>{int(max_tool_calls)}")
    for turn_key, constraint in (expected.get("turn_constraints") or {}).items():
        turn_calls = [call for call in calls if int(call.get("turn_index") or 0) == int(turn_key)]
        turn_names = {str(call.get("tool_name") or "") for call in turn_calls}
        denied = turn_names.intersection(str(name) for name in (constraint or {}).get("forbidden_tools") or [])
        if denied:
            failures.append(f"turn_{turn_key}_forbidden_tools:" + ",".join(sorted(denied)))

    dependency_failures = []
    for index, call in enumerate(calls):
        name = str(call.get("tool_name") or "")
        prior_names = {str(item.get("tool_name") or "") for item in calls[:index] if item.get("turn_index") == call.get("turn_index")}
        for dependency in dependencies.get(name) or []:
            if str(dependency) not in prior_names:
                dependency_failures.append(f"{name}<-{dependency}")
    if dependency_failures:
        failures.append("dependency_violation:" + ",".join(dependency_failures))

    provenance_failures = evaluate_argument_provenance(calls)
    failures.extend(provenance_failures)
    confirmation_failures = evaluate_confirmations(calls, confirmations)
    failures.extend(confirmation_failures)
    cross_user_failures = evaluate_cross_user_results(calls, user_id=user_id)
    failures.extend(cross_user_failures)

    required_coverage = 1.0 if not required else (len(required) - len(missing)) / len(required)
    dependency_score = 1.0 if not dependency_failures else 0.0
    provenance_score = 1.0 if not provenance_failures else 0.0
    confirmation_score = 1.0 if not confirmation_failures else 0.0
    scores = {
        "required_tool_coverage": round(required_coverage, 6),
        "tool_dependency_pass": dependency_score,
        "argument_provenance_pass": provenance_score,
        "human_confirmation_pass": confirmation_score,
        "forbidden_tool_pass": 1.0 if not present_forbidden else 0.0,
        "cross_user_isolation_pass": 1.0 if not cross_user_failures else 0.0,
        "tool_error_rate": round(sum(1 for call in calls if call.get("status") in {"tool_error", "invalid_arguments", "tool_not_found"}) / max(1, len(calls)), 6),
    }
    records = [score_record(name, value, failures) for name, value in scores.items()]
    return scores, failures, records


def evaluate_argument_provenance(calls: list[dict[str, Any]]) -> list[str]:
    failures = []
    for index, call in enumerate(calls):
        name = str(call.get("tool_name") or "")
        requirements = ARGUMENT_PROVENANCE.get(name) or {}
        prior = [item for item in calls[:index] if item.get("turn_index") == call.get("turn_index")]
        for argument_name, sources in requirements.items():
            actual_values = scalar_values((call.get("arguments") or {}).get(argument_name))
            if not actual_values:
                continue
            trusted = set()
            for source_tool, source_field in sources:
                for source_call in prior:
                    if source_call.get("tool_name") == source_tool:
                        trusted.update(find_field_values(source_call.get("result") or {}, source_field))
            if not actual_values.issubset(trusted):
                failures.append(f"untrusted_argument:{name}.{argument_name}")
    return failures


def evaluate_confirmations(calls: list[dict[str, Any]], required_tools: list[str]) -> list[str]:
    failures = []
    for call in calls:
        if call.get("tool_name") not in required_tools:
            continue
        result = call.get("result") or {}
        completed = any(
            value is True
            for value in find_field_values(result, "completed")
        )
        if call.get("ok") and not completed:
            failures.append(f"write_without_confirmation:{call.get('tool_name')}")
    return failures


def evaluate_cross_user_results(calls: list[dict[str, Any]], *, user_id: int) -> list[str]:
    for call in calls:
        values = find_field_values(call.get("result") or {}, "user_id")
        if any(str(value) not in {"", str(user_id)} for value in values):
            return [f"cross_user_result:{call.get('tool_name')}"]
    return []


def find_field_values(value: Any, field: str) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == field:
                found.update(scalar_values(child))
            found.update(find_field_values(child, field))
    elif isinstance(value, list):
        for child in value:
            found.update(find_field_values(child, field))
    return found


def scalar_values(value: Any) -> set[Any]:
    if isinstance(value, list):
        output: set[Any] = set()
        for child in value:
            output.update(scalar_values(child))
        return output
    if isinstance(value, dict):
        output = set()
        for child in value.values():
            output.update(scalar_values(child))
        return output
    if value is None or isinstance(value, (dict, list)):
        return set()
    try:
        hash(value)
        return {value}
    except TypeError:
        return {str(value)}


def score_record(name: str, value: float | None, failures: list[str]) -> dict[str, Any]:
    if value is None:
        return {
            "metric_name": name,
            "evaluator_type": "rule",
            "evaluator_version": "trajectory-v1",
            "score": None,
            "label": "not_applicable",
            "reason": "",
            "raw_result": {},
        }
    return {
        "metric_name": name,
        "evaluator_type": "rule",
        "evaluator_version": "trajectory-v1",
        "score": float(value),
        "label": "pass" if value >= 1.0 else "fail",
        "reason": ";".join(failures),
        "raw_result": {},
    }
