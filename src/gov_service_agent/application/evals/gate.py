"""评测运行发布门禁。"""

from __future__ import annotations

from typing import Any


DEFAULT_THRESHOLDS = {
    "task_success_rate": 0.95,
    "tool_dependency_pass": 0.98,
    "argument_provenance_pass": 0.98,
    "retrieval_hit_at_k": 0.95,
    "faithfulness": 0.95,
    "refusal_accuracy": 0.98,
    "max_relative_regression": 0.02,
    "max_p95_latency_increase": 0.15,
}

HARD_FAILURE_PREFIXES = (
    "write_without_confirmation:",
    "cross_user_result:",
    "untrusted_argument:",
    "nonexistent_citation:",
)


def evaluate_gate(
    metrics: dict[str, Any],
    *,
    failure_reasons: list[str],
    thresholds: dict[str, float] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    hard = sorted({reason for reason in failure_reasons if reason.startswith(HARD_FAILURE_PREFIXES)})
    soft = []
    for name in (
        "task_success_rate",
        "tool_dependency_pass",
        "argument_provenance_pass",
        "retrieval_hit_at_k",
        "faithfulness",
        "refusal_accuracy",
    ):
        value = metric_value(metrics, name)
        if value is not None and value < effective[name]:
            soft.append(f"{name}:{value:.4f}<{effective[name]:.4f}")
    for item in (comparison or {}).get("regressions") or []:
        name = str(item.get("metric") or "")
        before = float(item.get("baseline") or 0)
        after = float(item.get("candidate") or 0)
        if "p95_latency" in name:
            relative = (after - before) / before if before > 0 else 0.0
            if relative > effective["max_p95_latency_increase"]:
                soft.append(f"baseline_regression:{name}:{relative:.4f}")
        elif before - after > effective["max_relative_regression"]:
            soft.append(f"baseline_regression:{name}:{before - after:.4f}")
    return {
        "passed": not hard and not soft,
        "hard_failures": hard,
        "soft_failures": soft,
        "thresholds": effective,
    }


def metric_value(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    for section in ("rules", "ragas", "judge", "performance"):
        nested = metrics.get(section) or {}
        if section == "ragas":
            nested = nested.get("scores") or nested
        value = nested.get(name) if isinstance(nested, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return None
