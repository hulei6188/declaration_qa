"""评测数据集和运行路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from declaration_qa.api.auth import RequestContext, require_service_or_user
from declaration_qa.api.schemas import (
    EvaluationAnnotationRequest,
    EvaluationCaseRequest,
    EvaluationCaseResponse,
    EvaluationDatasetRequest,
    EvaluationDatasetResponse,
    EvaluationRunItemResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
)
from declaration_qa.api.server.dependencies import auth_context
from declaration_qa.api.server.knowledge_admin import ensure_knowledge_base_fresh
from declaration_qa.api.server.runtime import settings, workflow
from declaration_qa.application.evaluation import run_evaluation
from declaration_qa.application.evaluation_ragas import create_ragas_evaluator


def register_evaluation_routes(app):
    """注册评测数据集和运行路由。"""
    @app.post("/api/declaration-qa/evaluation/datasets", response_model=EvaluationDatasetResponse)
    def create_evaluation_dataset(request: EvaluationDatasetRequest, context: RequestContext = Depends(auth_context)):
        """创建对应业务记录并返回结果。"""
        require_service_or_user(context, settings)
        try:
            return workflow.audit_store.create_eval_dataset(
                name=request.name,
                description=request.description,
                cases=[case.model_dump(mode="json") for case in request.cases],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/declaration-qa/evaluation/datasets", response_model=list[EvaluationDatasetResponse])
    def list_evaluation_datasets(context: RequestContext = Depends(auth_context)):
        """查询评测数据集列表。"""
        require_service_or_user(context, settings)
        return workflow.audit_store.list_eval_datasets()

    @app.post("/api/declaration-qa/evaluation/datasets/{dataset_id}/cases", response_model=EvaluationCaseResponse)
    def add_evaluation_case(dataset_id: str, request: EvaluationCaseRequest, context: RequestContext = Depends(auth_context)):
        """新增评测用例。"""
        require_service_or_user(context, settings)
        try:
            return workflow.audit_store.add_eval_case(dataset_id=dataset_id, case=request.model_dump(mode="json"))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/declaration-qa/evaluation/datasets/{dataset_id}/cases", response_model=list[EvaluationCaseResponse])
    def list_evaluation_cases(dataset_id: str, context: RequestContext = Depends(auth_context)):
        """查询评测用例列表。"""
        require_service_or_user(context, settings)
        return workflow.audit_store.list_eval_cases(dataset_id)

    @app.post("/api/declaration-qa/evaluation/datasets/{dataset_id}/runs", response_model=EvaluationRunResponse)
    def run_evaluation_dataset(dataset_id: str, request: EvaluationRunRequest, context: RequestContext = Depends(auth_context)):
        """运行评测数据集。"""
        require_service_or_user(context, settings)
        ensure_knowledge_base_fresh()
        try:
            use_ragas = settings.ragas_enabled if request.use_ragas is None else request.use_ragas
            ragas_evaluator, unavailable_reason = (
                create_ragas_evaluator(workflow, settings)
                if use_ragas
                else (None, "")
            )
            return run_evaluation(
                workflow=workflow,
                store=workflow.audit_store,
                dataset_id=dataset_id,
                note=request.note,
                top_k=request.top_k,
                ragas_evaluator=ragas_evaluator,
                ragas_requested=use_ragas,
                ragas_unavailable_reason=unavailable_reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/declaration-qa/evaluation/runs/{run_id}", response_model=EvaluationRunResponse)
    def get_evaluation_run(run_id: str, context: RequestContext = Depends(auth_context)):
        """查询评测运行结果。"""
        require_service_or_user(context, settings)
        item = workflow.audit_store.get_eval_run(run_id)
        if not item:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        return item

    @app.post("/api/declaration-qa/evaluation/run-items/{item_id}/annotation", response_model=EvaluationRunItemResponse)
    def annotate_evaluation_item(item_id: str, request: EvaluationAnnotationRequest, context: RequestContext = Depends(auth_context)):
        """标注评测运行明细。"""
        require_service_or_user(context, settings)
        try:
            return workflow.audit_store.annotate_eval_item(item_id=item_id, label=request.label, comment=request.comment)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
