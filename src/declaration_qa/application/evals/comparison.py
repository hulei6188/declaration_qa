"""基线运行与候选运行比较。"""

from __future__ import annotations

from typing import Any


def compare_runs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = flatten_numeric(baseline.get("metrics") or {})
    candidate_metrics = flatten_numeric(candidate.get("metrics") or {})
    deltas = {}
    regressions = []
    improvements = []
    for name in sorted(set(baseline_metrics).intersection(candidate_metrics)):
        before = baseline_metrics[name]
        after = candidate_metrics[name]
        delta = round(after - before, 6)
        deltas[name] = {"baseline": before, "candidate": after, "delta": delta}
        lower_is_better = any(token in name for token in ("latency", "cost", "error_rate", "duplicate_rate", "token"))
        regressed = delta > 0 if lower_is_better else delta < 0
        improved = delta < 0 if lower_is_better else delta > 0
        target = {"metric": name, "baseline": before, "candidate": after, "delta": delta}
        if regressed:
            regressions.append(target)
        elif improved:
            improvements.append(target)
    return {
        "baseline_run_id": baseline.get("run_id") or "",
        "candidate_run_id": candidate.get("run_id") or "",
        "metric_deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
    }


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_numeric(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    return output
