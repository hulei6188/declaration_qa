"""线上Trace采样、Langfuse评分与失败候选自动回灌。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.application.evals.candidates import (
    candidate_payload,
    sample_production_traces,
)
from gov_service_agent.application.evals.judge import IndependentJudge
from gov_service_agent.application.evals.pii import sanitize_text
from gov_service_agent.application.factory import create_workflow
from gov_service_agent.core.config import GovServiceAgentSettings, get_settings
from gov_service_agent.observability.langfuse import observation, score_trace
from gov_service_agent.rag.model_providers.factories import build_agent_model


def run_online_evaluation_cycle(
    settings: GovServiceAgentSettings | None = None,
    *,
    workflow=None,
    judge: IndependentJudge | None = None,
) -> dict[str, Any]:
    """执行一个完整周期；设计为可由APScheduler或运维命令重复调用。"""
    settings = settings or get_settings()
    if not settings.online_evaluation_enabled:
        return {"status": "disabled", "sampled": 0, "scored": 0, "backfilled": 0}
    owns_workflow = workflow is None
    workflow = workflow or create_workflow(settings)
    try:
        judge = judge or build_online_judge(settings)
        samples = sample_production_traces(
            workflow.audit_store,
            salt=settings.evaluation_pii_salt,
            sample_rate=settings.online_evaluation_sample_rate,
            lookback_hours=settings.online_evaluation_lookback_hours,
            limit=settings.online_evaluation_sample_limit,
            include_failures=True,
        )
        scored = 0
        backfilled = 0
        low_faithfulness = 0
        errors: list[str] = []
        with observation(
            "evaluation.online_sampling_cycle",
            settings=settings,
            input={
                "sample_rate": settings.online_evaluation_sample_rate,
                "lookback_hours": settings.online_evaluation_lookback_hours,
                "sample_limit": settings.online_evaluation_sample_limit,
            },
            metadata={"component": "online_evaluation"},
            tags=["online-evaluation", "scheduled"],
            trace_name="online-evaluation-cycle",
        ) as cycle_span:
            for sample in samples:
                judged = judge.evaluate(
                    query=sanitize_text(str(sample.get("query") or ""), salt=settings.evaluation_pii_salt),
                    answer=sanitize_text(str(sample.get("answer") or ""), salt=settings.evaluation_pii_salt),
                    contexts=source_contexts(sample.get("sources") or []),
                    reference_answer="",
                )
                scores = dict(judged.get("scores") or {})
                if scores:
                    scored += 1
                faithfulness = scores.get("faithfulness")
                if isinstance(faithfulness, (int, float)) and float(faithfulness) < settings.online_evaluation_faithfulness_threshold:
                    low_faithfulness += 1
                trace_id = str(sample.get("trace_id") or "")
                for name, value in scores.items():
                    if isinstance(value, (int, float)):
                        score_trace(
                            trace_id,
                            name=f"online.{name}",
                            value=float(value),
                            comment="scheduled production trace evaluation",
                            settings=settings,
                        )
                score_trace(
                    trace_id,
                    name="online.sampled",
                    value=1,
                    comment="selected by stable production trace sampler",
                    settings=settings,
                )
                try:
                    workflow.audit_store.create_eval_candidate(
                        candidate_payload(
                            sample,
                            online_scores={
                                "scores": scores,
                                "status": dict(judged.get("status") or {}),
                            },
                            faithfulness_threshold=settings.online_evaluation_faithfulness_threshold,
                        )
                    )
                    backfilled += 1
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
            result = {
                "status": "success" if not errors else "partial",
                "sampled": len(samples),
                "scored": scored,
                "low_faithfulness": low_faithfulness,
                "backfilled": backfilled,
                "errors": errors[:20],
            }
            cycle_span.update(output=result)
            return result
    finally:
        if owns_workflow:
            workflow.close()


def build_online_judge(settings: GovServiceAgentSettings) -> IndependentJudge:
    if not settings.evaluation_judge_enabled:
        return IndependentJudge(None)
    model = build_agent_model(
        settings,
        timeout_seconds=settings.agent_timeout_seconds,
        model_name=settings.evaluation_judge_model,
        api_key=settings.evaluation_judge_api_key,
        base_url=settings.evaluation_judge_api_base,
    )
    return IndependentJudge(model, model_name=settings.evaluation_judge_model)


def source_contexts(sources: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    output = []
    seen = set()
    for source in sources[:limit]:
        text = str(source.get("evidence_text") or source.get("snippet") or source.get("content") or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text[:8000])
    return output
