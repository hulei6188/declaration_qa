"""生产评测数据集、运行、候选样本和发布门禁路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query, Response, status

from declaration_qa.api.auth import RequestContext, require_evaluation_admin
from declaration_qa.api.schemas import (
    EvaluationAnnotationRequest,
    EvaluationCandidatePromoteRequest,
    EvaluationCandidateResponse,
    EvaluationCandidateReviewRequest,
    EvaluationCaseRequest,
    EvaluationCaseResponse,
    EvaluationComparisonResponse,
    EvaluationDatasetRequest,
    EvaluationDatasetResponse,
    EvaluationGateResponse,
    EvaluationRunItemResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
)
from declaration_qa.api.server.dependencies import auth_context
from declaration_qa.api.server.knowledge_admin import ensure_knowledge_base_fresh
from declaration_qa.api.server.runtime import evaluation_queue, settings, workflow
from declaration_qa.application.evals.candidates import mine_candidates
from declaration_qa.application.evals.comparison import compare_runs


def register_evaluation_routes(app):
    @app.post("/api/declaration-qa/evaluation/datasets", response_model=EvaluationDatasetResponse)
    def create_evaluation_dataset(request: EvaluationDatasetRequest, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.create_eval_dataset(
                name=request.name,
                description=request.description,
                cases=[case.model_dump(mode="json") for case in request.cases],
                version=request.version,
                status=request.status,
                source=request.source,
                parent_dataset_id=request.parent_dataset_id,
                knowledge_revision=request.knowledge_revision,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/declaration-qa/evaluation/datasets", response_model=list[EvaluationDatasetResponse])
    def list_evaluation_datasets(context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        return workflow.audit_store.list_eval_datasets()

    @app.post("/api/declaration-qa/evaluation/datasets/{dataset_id}/publish", response_model=EvaluationDatasetResponse)
    def publish_evaluation_dataset(dataset_id: str, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.publish_eval_dataset(dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/declaration-qa/evaluation/datasets/{dataset_id}/cases", response_model=EvaluationCaseResponse)
    def add_evaluation_case(dataset_id: str, request: EvaluationCaseRequest, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.add_eval_case(dataset_id=dataset_id, case=request.model_dump(mode="json"))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/declaration-qa/evaluation/datasets/{dataset_id}/cases", response_model=list[EvaluationCaseResponse])
    def list_evaluation_cases(dataset_id: str, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        return workflow.audit_store.list_eval_cases(dataset_id)

    @app.post("/api/declaration-qa/evaluation/datasets/{dataset_id}/runs", response_model=EvaluationRunResponse)
    def run_evaluation_dataset(
        dataset_id: str,
        request: EvaluationRunRequest,
        response: Response,
        context: RequestContext = Depends(auth_context),
    ):
        require_evaluation_admin(context, settings)
        ensure_knowledge_base_fresh()
        use_ragas = settings.ragas_enabled if request.use_ragas is None else request.use_ragas
        snapshot = runtime_snapshot()
        try:
            run = workflow.audit_store.create_eval_run(
                dataset_id=dataset_id,
                note=request.note,
                top_k=request.top_k,
                baseline_run_id=request.baseline_run_id,
                trigger_source=request.trigger_source,
                runtime_snapshot=snapshot,
                git_commit=request.git_commit,
                status="queued",
            )
            queued = evaluation_queue.enqueue(
                run["run_id"],
                {
                    "dataset_id": dataset_id,
                    "top_k": request.top_k,
                    "use_ragas": use_ragas,
                },
            )
            if queued.get("backend") == "inline":
                return queued["result"]
            response.status_code = status.HTTP_202_ACCEPTED
            return workflow.audit_store.get_eval_run(run["run_id"])
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/declaration-qa/evaluation/runs/{run_id}", response_model=EvaluationRunResponse)
    def get_evaluation_run(run_id: str, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        item = workflow.audit_store.get_eval_run(run_id)
        if not item:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        return item

    @app.get(
        "/api/declaration-qa/evaluation/datasets/{dataset_id}/runs",
        response_model=list[EvaluationRunResponse],
    )
    def list_evaluation_runs(
        dataset_id: str,
        limit: int = Query(default=50, ge=1, le=200),
        context: RequestContext = Depends(auth_context),
    ):
        require_evaluation_admin(context, settings)
        return workflow.audit_store.list_eval_runs(dataset_id=dataset_id, limit=limit)

    @app.get(
        "/api/declaration-qa/evaluation/runs/{run_id}/compare/{baseline_run_id}",
        response_model=EvaluationComparisonResponse,
    )
    def compare_evaluation_runs(run_id: str, baseline_run_id: str, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        candidate = workflow.audit_store.get_eval_run(run_id)
        baseline = workflow.audit_store.get_eval_run(baseline_run_id)
        if not candidate or not baseline:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        return compare_runs(baseline, candidate)

    @app.get("/api/declaration-qa/evaluation/runs/{run_id}/gate", response_model=EvaluationGateResponse)
    def get_evaluation_gate(run_id: str, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        run = workflow.audit_store.get_eval_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        gate = dict(run.get("gate_result") or {})
        return {"run_id": run_id, "passed": bool(gate.get("passed")), **gate}

    @app.post("/api/declaration-qa/evaluation/run-items/{item_id}/annotation", response_model=EvaluationRunItemResponse)
    def annotate_evaluation_item(item_id: str, request: EvaluationAnnotationRequest, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.annotate_eval_item(item_id=item_id, label=request.label, comment=request.comment)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/declaration-qa/evaluation/candidates/mine", response_model=list[EvaluationCandidateResponse])
    def mine_evaluation_candidates(context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        return mine_candidates(
            workflow.audit_store,
            salt=settings.evaluation_pii_salt,
            limit=settings.evaluation_candidate_scan_limit,
        )

    @app.get("/api/declaration-qa/evaluation/candidates", response_model=list[EvaluationCandidateResponse])
    def list_evaluation_candidates(
        candidate_status: str = Query(default="", alias="status"),
        limit: int = Query(default=100, ge=1, le=500),
        context: RequestContext = Depends(auth_context),
    ):
        require_evaluation_admin(context, settings)
        return workflow.audit_store.list_eval_candidates(status=candidate_status, limit=limit)

    @app.post("/api/declaration-qa/evaluation/candidates/{candidate_id}/review", response_model=EvaluationCandidateResponse)
    def review_evaluation_candidate(
        candidate_id: str,
        request: EvaluationCandidateReviewRequest,
        context: RequestContext = Depends(auth_context),
    ):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.review_eval_candidate(
                candidate_id=candidate_id,
                status=request.status,
                severity=request.severity,
                labels=request.labels,
                comment=request.comment,
                reviewer_id=int(context.user_id or 0),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/declaration-qa/evaluation/candidates/{candidate_id}/promote", response_model=EvaluationCaseResponse)
    def promote_evaluation_candidate(
        candidate_id: str,
        request: EvaluationCandidatePromoteRequest,
        context: RequestContext = Depends(auth_context),
    ):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.promote_eval_candidate(
                candidate_id=candidate_id,
                dataset_id=request.dataset_id,
                case=request.expected.model_dump(mode="json"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def runtime_snapshot() -> dict:
    revision = ""
    revision_reader = getattr(workflow.audit_store, "knowledge_base_revision", None)
    if callable(revision_reader):
        revision = str(revision_reader() or "")
    return {
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "agent_model": settings.agent_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model,
        "rerank_score_threshold": settings.rerank_score_threshold,
        "retrieval_candidates_per_query": settings.retrieval_candidates_per_query,
        "rerank_top_n_per_query": settings.rerank_top_n_per_query,
        "agent_max_iterations": settings.agent_max_iterations,
        "knowledge_revision": revision,
        "prompt_version": "single-agent-tool-loop-v1",
        "tool_registry_version": "tool-registry-v1",
    }
