"""RQ评测任务入口。"""

from __future__ import annotations

from gov_service_agent.application.evaluation_ragas import create_ragas_evaluator
from gov_service_agent.application.evals.judge import IndependentJudge
from gov_service_agent.application.evals.runner import EvaluationRunner
from gov_service_agent.application.factory import create_evaluation_workflow, create_workflow
from gov_service_agent.core.config import get_settings
from gov_service_agent.rag.model_providers.factories import build_agent_model


def process_evaluation_run_task(run_id: str, payload: dict) -> dict:
    """在独立评测工作流中执行已经持久化的运行。"""
    settings = get_settings()
    primary_workflow = create_workflow(settings)
    evaluation_workflow = create_evaluation_workflow(settings)
    try:
        use_ragas = bool(payload.get("use_ragas"))
        ragas_evaluator, unavailable_reason = (
            create_ragas_evaluator(evaluation_workflow, settings) if use_ragas else (None, "")
        )
        judge_model = None
        if settings.evaluation_judge_enabled:
            judge_model = build_agent_model(
                settings,
                timeout_seconds=settings.agent_timeout_seconds,
                model_name=settings.evaluation_judge_model,
                api_key=settings.evaluation_judge_api_key,
                base_url=settings.evaluation_judge_api_base,
            )
        runner = EvaluationRunner(
            workflow=evaluation_workflow,
            store=primary_workflow.audit_store,
            ragas_evaluator=ragas_evaluator,
            ragas_requested=use_ragas,
            ragas_unavailable_reason=unavailable_reason,
            judge=IndependentJudge(judge_model, model_name=settings.evaluation_judge_model)
            if settings.evaluation_judge_enabled
            else None,
            langfuse_settings=settings,
            run_retrieval_ablation_enabled=bool(payload.get("run_retrieval_ablation")),
        )
        return runner.run(
            dataset_id=str(payload["dataset_id"]),
            run_id=run_id,
            top_k=int(payload.get("top_k") or 4),
        )
    finally:
        evaluation_workflow.close()
        primary_workflow.close()
