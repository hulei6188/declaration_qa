"""服务运行时组件。"""

from __future__ import annotations

from atexit import register

from declaration_qa.application.factory import create_index_job_store, create_workflow
from declaration_qa.core.config import get_settings
from declaration_qa.core.runtime import assert_production_ready
from declaration_qa.indexing.queue import build_index_queue
from declaration_qa.indexing.service.processor import process_index_job
from declaration_qa.application.evals.judge import IndependentJudge
from declaration_qa.application.evals.queue import build_evaluation_queue
from declaration_qa.application.evals.runner import EvaluationRunner
from declaration_qa.application.evaluation_ragas import create_ragas_evaluator
from declaration_qa.application.factory import create_evaluation_workflow
from declaration_qa.rag.model_providers.factories import build_agent_model


settings = get_settings()
assert_production_ready(settings)
workflow = create_workflow(settings)
register(workflow.close)
index_job_store = create_index_job_store(workflow)


def process_index_job_inline(job_uid: str) -> dict:
    """以内联方式处理索引任务。"""
    return process_index_job(job_uid, workflow=workflow, job_store=index_job_store, settings=settings)


index_queue = build_index_queue(settings, inline_handler=process_index_job_inline, job_store=index_job_store)


def process_evaluation_inline(run_id: str, payload: dict) -> dict:
    """开发模式内联执行；生产模式由RQ Worker执行。"""
    evaluation_workflow = create_evaluation_workflow(settings) if settings.effective_evaluation_database_url else workflow
    owns_workflow = evaluation_workflow is not workflow
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
            store=workflow.audit_store,
            ragas_evaluator=ragas_evaluator,
            ragas_requested=use_ragas,
            ragas_unavailable_reason=unavailable_reason,
            judge=IndependentJudge(judge_model, model_name=settings.evaluation_judge_model)
            if settings.evaluation_judge_enabled
            else None,
            langfuse_settings=settings,
        )
        return runner.run(
            dataset_id=str(payload["dataset_id"]),
            run_id=run_id,
            top_k=int(payload.get("top_k") or 4),
        )
    finally:
        if owns_workflow:
            evaluation_workflow.close()


evaluation_queue = build_evaluation_queue(settings, inline_handler=process_evaluation_inline)
