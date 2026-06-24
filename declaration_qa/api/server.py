"""FastAPI 应用创建、接口路由和流式响应实现。"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Iterable
from dataclasses import replace
from datetime import date
import re
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from declaration_qa import __version__
from declaration_qa.storage.audit import InMemoryAuditStore
from declaration_qa.api.auth import RequestContext, apply_chat_auth, require_service_or_user, resolve_request_context
from declaration_qa.core.config import get_settings
from declaration_qa.application.evaluation import run_evaluation
from declaration_qa.application.factory import create_index_job_store, create_workflow
from declaration_qa.indexing.service import process_index_job
from declaration_qa.indexing.queue import build_index_queue
from declaration_qa.rag.engine import normalize_chunking_strategy
from declaration_qa.rag.retrieval import sanitize_metadata
from declaration_qa.core.runtime import assert_production_ready, production_readiness_errors
from declaration_qa.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteDocumentResponse,
    DocumentCompareResponse,
    DocumentRollbackRequest,
    DocumentRolloutRequest,
    DocumentRolloutResponse,
    DocumentStatusResponse,
    DocumentVersionRequest,
    DocumentVersionResponse,
    DeleteSessionResponse,
    EvaluationAnnotationRequest,
    EvaluationCaseRequest,
    EvaluationCaseResponse,
    EvaluationDatasetRequest,
    EvaluationDatasetResponse,
    EvaluationRunItemResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    FeedbackRequest,
    IndexJobResponse,
    KnowledgeDocumentAdminRecord,
    KnowledgeDocumentRequest,
    KnowledgeBatchUploadResponse,
    KnowledgeDocumentResponse,
    KnowledgeUploadResponse,
    ReindexResponse,
    RetryIndexJobResponse,
    SessionDetail,
    SessionSummary,
    TicketSummary,
)
from declaration_qa.application.workflow import DeclarationQAWorkflow


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


settings = get_settings()
assert_production_ready(settings)
workflow = create_workflow(settings)
index_job_store = create_index_job_store(workflow)
_knowledge_sync_lock = Lock()
try:
    _knowledge_base_revision = (
        workflow.audit_store.knowledge_base_revision()
        if hasattr(workflow.audit_store, "knowledge_base_revision")
        else ""
    )
except Exception:
    _knowledge_base_revision = ""


def process_index_job_inline(job_uid: str) -> dict:
    """以内联方式处理索引任务。"""
    return process_index_job(job_uid, workflow=workflow, job_store=index_job_store, settings=settings)


index_queue = build_index_queue(settings, inline_handler=process_index_job_inline, job_store=index_job_store)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    def auth_context(request: Request) -> RequestContext:
        """解析当前请求的认证上下文。"""
        return resolve_request_context(request, settings)

    app = FastAPI(
        title="Declaration QA Workflow",
        version=__version__,
        description="Isolated FastAPI + LangGraph declaration QA service.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5174",
            "http://localhost:5174",
            "http://127.0.0.1:8020",
            "http://localhost:8020",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/declaration-qa/health")
    def health():
        """返回服务健康检查信息。"""
        return {
            "ok": True,
            "service": "declaration-qa",
            "entrypoint": "declaration_qa.api.server:app",
            "chunks": workflow.rag_engine.chunk_count,
            "cache_backend": workflow.rag_engine.cache_store.backend,
            "vector_backend": workflow.rag_engine.vector_index.backend,
            "keyword_backend": getattr(workflow.rag_engine.keyword_index, "backend", "memory"),
            "external_chunk_store": bool(workflow.rag_engine.chunk_store),
            "business_adapter": workflow.business_adapter.__class__.__name__,
            "audit_store": workflow.audit_store.__class__.__name__,
            "embedding_provider": workflow.rag_engine.embedding_provider.name,
            "rerank_provider": workflow.rag_engine.reranker.name,
            "question_planner": getattr(workflow.question_planner, "name", workflow.question_planner.__class__.__name__),
            "query_rewriter": getattr(workflow.query_rewriter, "name", workflow.query_rewriter.__class__.__name__),
            "answer_generator": workflow.answer_generator.__class__.__name__,
            "max_sub_question_workers": workflow.max_sub_question_workers,
            "max_retrieval_workers": workflow.max_retrieval_workers,
            "auth_required": settings.effective_auth_required,
            "production_ready": not production_readiness_errors(settings),
            "production_readiness_errors": production_readiness_errors(settings),
            "index_queue_backend": index_queue.backend,
        }

    @app.post("/api/declaration-qa/admin/init-schema")
    def init_schema():
        """初始化数据库表结构。"""
        if not hasattr(workflow.audit_store, "init_schema"):
            return {"initialized": False, "message": "PostgreSQL store is not configured"}
        workflow.audit_store.init_schema()
        return {"initialized": True}

    @app.post("/api/declaration-qa/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, context: RequestContext = Depends(auth_context)):
        """处理非流式问答请求。"""
        ensure_knowledge_base_fresh()
        return workflow.invoke(apply_chat_auth(request, context, settings))

    @app.post("/api/declaration-qa/chat/stream")
    def chat_stream(request: ChatRequest, context: RequestContext = Depends(auth_context)):
        """处理流式问答请求。"""
        ensure_knowledge_base_fresh()
        return StreamingResponse(
            sse_event_stream(workflow.stream(apply_chat_auth(request, context, settings))),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.get("/api/declaration-qa/tickets", response_model=list[TicketSummary])
    def list_tickets(
        user_id: int = 1,
        biz_type: str = "",
        region_code: str = "340000",
        context: RequestContext = Depends(auth_context),
    ):
        """查询当前用户办件列表。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        if not hasattr(workflow.business_adapter, "list_applications"):
            return []
        return workflow.business_adapter.list_applications(
            user_id=int(effective_user_id or 1),
            biz_type=biz_type or None,
            region_code=region_code or None,
        )

    @app.post("/api/declaration-qa/knowledge/documents", response_model=KnowledgeDocumentResponse)
    def add_document(request: KnowledgeDocumentRequest, context: RequestContext = Depends(auth_context)):
        """创建文本知识文档索引任务。"""
        require_service_or_user(context, settings)
        document_id = f"doc-{uuid4().hex}"
        job = index_job_store.create_index_job(
            job_type="document_index",
            payload={
                "input_type": "text",
                "document_id": document_id,
                "title": request.title,
                "content": request.content,
                "filename": "",
                "content_type": "text/plain",
                "chunking_strategy": normalize_chunking_strategy(request.chunking_strategy),
                "region_code": request.region_code,
                "policy_version": request.policy_version,
                "source_url": request.source_url,
                "audit_node": request.audit_node,
                "status": request.status,
                "effective_start_time": request.effective_start_time.isoformat() if request.effective_start_time else None,
                "effective_end_time": request.effective_end_time.isoformat() if request.effective_end_time else None,
                "metadata": request.metadata,
            },
        )
        index_queue.enqueue(job["job_uid"])
        final_job = index_job_store.get_index_job(job["job_uid"]) or job
        final_payload = final_job.get("payload") or {}
        return KnowledgeDocumentResponse(
            document_id=document_id,
            chunks_indexed=int(final_payload.get("chunks_indexed") or 0),
            job_id=job["job_uid"],
        )

    @app.post("/api/declaration-qa/knowledge/upload", response_model=KnowledgeUploadResponse)
    async def upload_document(
        context: RequestContext = Depends(auth_context),
        file: UploadFile = File(...),
        title: str = Form(""),
        region_code: str = Form("340000"),
        policy_version: str = Form("2025"),
        source_url: str = Form(""),
        audit_node: str = Form(""),
        status: str = Form("published"),
        effective_start_time: str = Form(""),
        effective_end_time: str = Form(""),
        metadata: str = Form(""),
        chunking_strategy: str = Form("parent_child"),
    ):
        """接收单个文件上传并创建索引任务。"""
        require_service_or_user(context, settings)
        upload_data = await file.read()
        return enqueue_upload_document(
            data=upload_data,
            filename=file.filename or "upload.bin",
            content_type=file.content_type or "application/octet-stream",
            title=title,
            region_code=region_code,
            policy_version=policy_version,
            source_url=source_url,
            audit_node=audit_node,
            status=status,
            effective_start_time=effective_start_time,
            effective_end_time=effective_end_time,
            metadata=parse_metadata_form(metadata),
            chunking_strategy=chunking_strategy,
        )

    @app.post("/api/declaration-qa/knowledge/uploads", response_model=KnowledgeBatchUploadResponse)
    async def upload_documents(
        context: RequestContext = Depends(auth_context),
        files: list[UploadFile] = File(...),
        title: str = Form(""),
        region_code: str = Form("340000"),
        policy_version: str = Form("2025"),
        source_url: str = Form(""),
        audit_node: str = Form(""),
        status: str = Form("published"),
        effective_start_time: str = Form(""),
        effective_end_time: str = Form(""),
        metadata: str = Form(""),
        chunking_strategy: str = Form("parent_child"),
    ):
        """接收批量文件上传并创建索引任务。"""
        require_service_or_user(context, settings)
        parsed_metadata = parse_metadata_form(metadata)
        responses = []
        for file_index, upload in enumerate(files):
            upload_data = await upload.read()
            filename = upload.filename or f"upload-{file_index + 1}.bin"
            item_title = batch_upload_title(title, filename, len(files))
            responses.append(
                enqueue_upload_document(
                    data=upload_data,
                    filename=filename,
                    content_type=upload.content_type or "application/octet-stream",
                    title=item_title,
                    region_code=region_code,
                    policy_version=policy_version,
                    source_url=source_url,
                    audit_node=audit_node,
                    status=status,
                    effective_start_time=effective_start_time,
                    effective_end_time=effective_end_time,
                    metadata=parsed_metadata,
                    chunking_strategy=chunking_strategy,
                )
            )
        return KnowledgeBatchUploadResponse(total_files=len(responses), documents=responses)

    @app.post("/api/declaration-qa/knowledge/reindex", response_model=ReindexResponse)
    def reindex_documents(context: RequestContext = Depends(auth_context)):
        """创建知识库重建索引任务。"""
        require_service_or_user(context, settings)
        job = index_job_store.create_index_job(job_type="knowledge_base_reindex", payload={})
        index_queue.enqueue(job["job_uid"])
        final_job = index_job_store.get_index_job(job["job_uid"]) or job
        final_payload = final_job.get("payload") or {}
        return ReindexResponse(
            job_id=job["job_uid"],
            documents_indexed=int(final_payload.get("documents_indexed") or 0),
            chunks_indexed=int(final_payload.get("chunks_indexed") or 0),
        )

    @app.get("/api/declaration-qa/index-jobs/{job_id}", response_model=IndexJobResponse)
    def get_index_job(job_id: str, context: RequestContext = Depends(auth_context)):
        """查询索引任务状态。"""
        require_service_or_user(context, settings)
        item = index_job_store.get_index_job(job_id)
        if not item:
            raise HTTPException(status_code=404, detail="Index job not found")
        return item

    @app.post("/api/declaration-qa/index-jobs/{job_id}/retry", response_model=RetryIndexJobResponse)
    def retry_index_job(job_id: str, context: RequestContext = Depends(auth_context)):
        """基于失败任务创建重试任务。"""
        require_service_or_user(context, settings)
        item = index_job_store.get_index_job(job_id)
        if not item:
            raise HTTPException(status_code=404, detail="Index job not found")
        payload = item.get("payload") or {}
        job_type = item.get("job_type")
        if job_type not in {"knowledge_base_reindex", "document_delete", "document_index"}:
            raise HTTPException(status_code=400, detail=f"Unsupported index job type: {job_type}")
        index_queue.enqueue(job_id)
        updated = index_job_store.get_index_job(job_id) or item
        return {**updated, "retried": True}

    @app.delete("/api/declaration-qa/knowledge/documents/{document_id}", response_model=DeleteDocumentResponse)
    def delete_document(document_id: str, context: RequestContext = Depends(auth_context)):
        """删除知识文档及相关片段。"""
        require_service_or_user(context, settings)
        job = index_job_store.create_index_job(job_type="document_delete", payload={"document_id": document_id})
        index_queue.enqueue(job["job_uid"])
        final_job = index_job_store.get_index_job(job["job_uid"]) or job
        final_payload = final_job.get("payload") or {}
        return DeleteDocumentResponse(document_id=document_id, deleted=bool(final_payload.get("deleted")), job_id=job["job_uid"])

    @app.get("/api/declaration-qa/knowledge/documents", response_model=list[KnowledgeDocumentAdminRecord])
    def list_knowledge_documents(status: str | None = None, context: RequestContext = Depends(auth_context)):
        """查询知识文档管理列表。"""
        require_service_or_user(context, settings)
        return list_document_records(status=status)

    @app.post("/api/declaration-qa/knowledge/documents/{document_id}/publish", response_model=DocumentStatusResponse)
    def publish_document(document_id: str, context: RequestContext = Depends(auth_context)):
        """发布指定知识文档。"""
        require_service_or_user(context, settings)
        payload = set_document_status(document_id, "published")
        ensure_knowledge_base_fresh(force=True)
        return DocumentStatusResponse(document_id=document_id, status=payload["status"])

    @app.post("/api/declaration-qa/knowledge/documents/{document_id}/unpublish", response_model=DocumentStatusResponse)
    def unpublish_document(document_id: str, context: RequestContext = Depends(auth_context)):
        """下线指定知识文档。"""
        require_service_or_user(context, settings)
        payload = set_document_status(document_id, "deprecated")
        ensure_knowledge_base_fresh(force=True)
        return DocumentStatusResponse(document_id=document_id, status=payload["status"])

    @app.post("/api/declaration-qa/knowledge/documents/{document_id}/rollout", response_model=DocumentRolloutResponse)
    def update_document_rollout(document_id: str, request: DocumentRolloutRequest, context: RequestContext = Depends(auth_context)):
        """调整知识文档灰度比例。"""
        require_service_or_user(context, settings)
        payload = update_rollout(document_id, request)
        ensure_knowledge_base_fresh(force=True)
        return DocumentRolloutResponse(**payload)

    @app.post("/api/declaration-qa/knowledge/documents/{document_id}/versions", response_model=DocumentVersionResponse)
    def create_document_version(document_id: str, request: DocumentVersionRequest, context: RequestContext = Depends(auth_context)):
        """基于已有文档创建新版本。"""
        require_service_or_user(context, settings)
        new_document_id = f"doc-{uuid4().hex}"
        payload = create_version(document_id, new_document_id, request)
        job = index_job_store.create_index_job(
            job_type="document_index",
            payload={
                "input_type": "text",
                "document_id": new_document_id,
                "title": payload["title"],
                "content": payload["content"],
                "filename": payload.get("filename", ""),
                "content_type": payload.get("content_type", "text/plain"),
                "region_code": payload["region_code"],
                "policy_version": payload["policy_version"],
                "source_url": payload.get("source_url", ""),
                "audit_node": payload.get("audit_node", ""),
                "status": "draft",
                "version_group_uid": payload["version_group_uid"],
                "version_no": payload["version_no"],
                "rollout_percent": 0,
                "change_note": request.change_note,
                "effective_start_time": payload.get("effective_start_time"),
                "effective_end_time": payload.get("effective_end_time"),
                "metadata": payload.get("metadata") or {},
            },
        )
        index_queue.enqueue(job["job_uid"])
        return DocumentVersionResponse(
            document_id=new_document_id,
            version_group_uid=payload["version_group_uid"],
            version_no=payload["version_no"],
            status="draft",
            job_id=job["job_uid"],
        )

    @app.get("/api/declaration-qa/knowledge/documents/{left_document_id}/compare/{right_document_id}", response_model=DocumentCompareResponse)
    def compare_documents(left_document_id: str, right_document_id: str, context: RequestContext = Depends(auth_context)):
        """比较两个知识文档版本的文本差异。"""
        require_service_or_user(context, settings)
        left = get_document_record(left_document_id)
        right = get_document_record(right_document_id)
        if not left or not right:
            raise HTTPException(status_code=404, detail="Document not found")
        added, removed = compare_text(str(left.get("raw_text") or ""), str(right.get("raw_text") or ""))
        return DocumentCompareResponse(
            left_document_id=left_document_id,
            right_document_id=right_document_id,
            added=added,
            removed=removed,
            changed=bool(added or removed),
        )

    @app.post("/api/declaration-qa/knowledge/documents/rollback", response_model=DocumentStatusResponse)
    def rollback_document(request: DocumentRollbackRequest, context: RequestContext = Depends(auth_context)):
        """将文档版本组回滚到指定版本。"""
        require_service_or_user(context, settings)
        payload = rollback_to_document(request.target_document_id, request.change_note)
        ensure_knowledge_base_fresh(force=True)
        return DocumentStatusResponse(document_id=payload["document_id"], status=payload["status"])

    @app.get("/api/declaration-qa/sessions", response_model=list[SessionSummary])
    def list_sessions(user_id: int = 1, context: RequestContext = Depends(auth_context)):
        """按用户查询历史会话摘要列表。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authenticated user_id is required")
        return workflow.audit_store.list_sessions(user_id=int(effective_user_id))

    @app.get("/api/declaration-qa/sessions/{session_id}", response_model=SessionDetail)
    def get_session(session_id: str, user_id: int = 1, context: RequestContext = Depends(auth_context)):
        """按用户和会话 ID 查询完整会话详情。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authenticated user_id is required")
        item = workflow.audit_store.get_session(user_id=int(effective_user_id), session_id=session_id)
        if not item:
            raise HTTPException(status_code=404, detail="Session not found")
        return item

    @app.delete("/api/declaration-qa/sessions/{session_id}", response_model=DeleteSessionResponse)
    def delete_session(session_id: str, user_id: int = 1, context: RequestContext = Depends(auth_context)):
        """按用户删除指定历史会话。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authenticated user_id is required")
        if not hasattr(workflow.audit_store, "delete_session"):
            raise HTTPException(status_code=501, detail="Session deletion is not supported")
        deleted = workflow.audit_store.delete_session(user_id=int(effective_user_id), session_id=session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return DeleteSessionResponse(session_id=session_id, deleted=True)

    @app.get("/api/declaration-qa/audit/{audit_id}")
    def get_audit(audit_id: str, context: RequestContext = Depends(auth_context)):
        """查询并返回对应的数据详情。"""
        require_service_or_user(context, settings)
        item = workflow.audit_store.get(audit_id)
        if not item:
            raise HTTPException(status_code=404, detail="Audit log not found")
        return item

    @app.post("/api/declaration-qa/feedback")
    def feedback(request: FeedbackRequest, context: RequestContext = Depends(auth_context)):
        """执行当前函数对应的业务处理。"""
        require_service_or_user(context, settings)
        if not workflow.audit_store.get(request.audit_id):
            raise HTTPException(status_code=404, detail="Audit log not found")
        return workflow.audit_store.add_feedback(
            audit_id=request.audit_id,
            rating=request.rating,
            comment=request.comment,
        )

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
            return run_evaluation(
                workflow=workflow,
                store=workflow.audit_store,
                dataset_id=dataset_id,
                note=request.note,
                top_k=request.top_k,
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

    return app


def parse_optional_date(value: str) -> date | None:
    """将可选输入解析为日期对象。"""
    if not value:
        return None
    return date.fromisoformat(value)


def ensure_knowledge_base_fresh(*, force: bool = False) -> None:
    """在需要时根据知识库修订刷新 RAG 索引。"""
    global _knowledge_base_revision
    if not hasattr(workflow.audit_store, "knowledge_base_revision"):
        return
    revision = workflow.audit_store.knowledge_base_revision()
    if not force and revision == _knowledge_base_revision:
        return
    with _knowledge_sync_lock:
        revision = workflow.audit_store.knowledge_base_revision()
        if not force and revision == _knowledge_base_revision:
            return
        if hasattr(workflow.audit_store, "list_indexed_knowledge") and hasattr(workflow.rag_engine, "load_indexed_data"):
            documents, chunks = workflow.audit_store.list_indexed_knowledge()
            workflow.rag_engine.load_indexed_data(documents, chunks)
        elif hasattr(workflow.audit_store, "list_documents"):
            documents = workflow.audit_store.list_documents()
            workflow.rag_engine.rebuild(documents)
        _knowledge_base_revision = revision


def list_document_records(status: str | None = None) -> list[dict]:
    """查询后台管理用知识文档记录。"""
    if hasattr(workflow.audit_store, "list_document_records"):
        return workflow.audit_store.list_document_records(status=status)
    records = []
    for document in workflow.rag_engine.documents():
        if status and document.status != status:
            continue
        records.append(document_record_from_domain(document))
    return records


def get_document_record(document_id: str) -> dict | None:
    """查询单条后台管理用知识文档记录。"""
    if hasattr(workflow.audit_store, "get_document_record"):
        item = workflow.audit_store.get_document_record(document_id)
        if item:
            return item
    for document in workflow.rag_engine.documents():
        if document.id == document_id:
            return {**document_record_from_domain(document), "raw_text": document.content, "content_type": document.content_type, "source_url": document.source_url}
    return None


def document_record_from_domain(document) -> dict:
    """将领域文档转换为后台记录。"""
    return {
        "document_id": document.id,
        "title": document.title,
        "filename": document.filename,
        "region_code": document.region_code,
        "policy_version": document.policy_version,
        "audit_node": document.audit_node,
        "status": document.status,
        "version_group_uid": document.version_group_uid or document.id,
        "version_no": document.version_no,
        "rollout_percent": document.rollout_percent,
        "effective_start_time": document.effective_start_time.isoformat() if document.effective_start_time else None,
        "effective_end_time": document.effective_end_time.isoformat() if document.effective_end_time else None,
        "chunk_count": len(workflow.rag_engine.document_chunks(document)),
        "change_note": document.change_note,
        "metadata": sanitize_metadata(document.metadata),
        "created_at": "",
        "updated_at": "",
    }


def parse_metadata_form(value: str) -> dict:
    """解析表单中的元数据 JSON。"""
    if not str(value or "").strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return sanitize_metadata(parsed)


def enqueue_upload_document(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    title: str,
    region_code: str,
    policy_version: str,
    source_url: str,
    audit_node: str,
    status: str,
    effective_start_time: str,
    effective_end_time: str,
    metadata: dict,
    chunking_strategy: str,
) -> KnowledgeUploadResponse:
    """创建上传文档索引任务并投递队列。"""
    document_id = stable_upload_document_id(data)
    staged_path = stage_upload_bytes(data, filename=filename, document_id=document_id)
    strategy = normalize_chunking_strategy(chunking_strategy)
    job = index_job_store.create_index_job(
        job_type="document_index",
        payload={
            "input_type": "file",
            "document_id": document_id,
            "title": title or filename or document_id,
            "filename": filename or staged_path.name,
            "content_type": content_type or "application/octet-stream",
            "file_path": str(staged_path),
            "chunking_strategy": strategy,
            "region_code": region_code,
            "policy_version": policy_version,
            "source_url": source_url,
            "audit_node": audit_node,
            "status": status,
            "effective_start_time": effective_start_time or None,
            "effective_end_time": effective_end_time or None,
            "metadata": sanitize_metadata(metadata),
        },
    )
    index_queue.enqueue(job["job_uid"])
    final_job = index_job_store.get_index_job(job["job_uid"]) or job
    final_payload = final_job.get("payload") or {}
    return KnowledgeUploadResponse(
        document_id=document_id,
        chunks_indexed=int(final_payload.get("chunks_indexed") or 0),
        job_id=job["job_uid"],
        filename=str(final_payload.get("filename") or filename or staged_path.name),
        content_type=str(final_payload.get("content_type") or content_type or "application/octet-stream"),
        chunking_strategy=str(final_payload.get("chunking_strategy") or strategy),
    )


def batch_upload_title(title: str, filename: str, file_count: int) -> str:
    """生成批量上传文档标题。"""
    if not title:
        return filename
    return f"{title} - {filename}" if file_count > 1 else title


def date_or_text(value) -> str | None:
    """将日期对象或文本转换为接口字符串。"""
    if not value:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def set_document_status(document_id: str, status: str) -> dict:
    """更新知识文档发布状态。"""
    if hasattr(workflow.audit_store, "set_document_status"):
        return workflow.audit_store.set_document_status(document_id, status)
    document = find_rag_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    workflow.rag_engine.delete_document(document_id)
    workflow.rag_engine.index_document(replace(document, status=status))
    return {"document_id": document_id, "status": status}


def update_rollout(document_id: str, request: DocumentRolloutRequest) -> dict:
    """更新文档灰度并刷新知识库。"""
    if hasattr(workflow.audit_store, "update_document_rollout"):
        item = workflow.audit_store.update_document_rollout(
            document_uid=document_id,
            rollout_percent=request.rollout_percent,
            effective_start_time=request.effective_start_time,
            effective_end_time=request.effective_end_time,
        )
        return {
            "document_id": str(item["document_id"]),
            "rollout_percent": int(item["rollout_percent"]),
            "effective_start_time": date_or_text(item.get("effective_start_time")),
            "effective_end_time": date_or_text(item.get("effective_end_time")),
        }
    document = find_rag_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    updated = replace(
        document,
        rollout_percent=request.rollout_percent,
        effective_start_time=request.effective_start_time,
        effective_end_time=request.effective_end_time,
    )
    workflow.rag_engine.delete_document(document_id)
    workflow.rag_engine.index_document(updated)
    return {
        "document_id": document_id,
        "rollout_percent": request.rollout_percent,
        "effective_start_time": date_or_text(request.effective_start_time),
        "effective_end_time": date_or_text(request.effective_end_time),
    }


def create_version(source_document_id: str, new_document_id: str, request: DocumentVersionRequest) -> dict:
    """创建文档新版本并刷新知识库。"""
    source = get_document_record(source_document_id)
    if not source:
        raise HTTPException(status_code=404, detail="Document not found")
    version_group_uid = str(source.get("version_group_uid") or source_document_id)
    existing_versions = [item["version_no"] for item in list_document_records() if item.get("version_group_uid") == version_group_uid]
    version_no = max(existing_versions or [int(source.get("version_no") or 1)]) + 1
    metadata = sanitize_metadata(dict(source.get("metadata") or {}))
    if request.policy_version:
        metadata["policy_version"] = request.policy_version
    return {
        "document_id": new_document_id,
        "version_group_uid": version_group_uid,
        "version_no": version_no,
        "title": request.title or str(source.get("title") or ""),
        "content": request.content if request.content is not None else str(source.get("raw_text") or ""),
        "filename": str(source.get("filename") or ""),
        "content_type": str(source.get("content_type") or "text/plain"),
        "region_code": str(metadata.get("region_code") or metadata.get("region_key") or source.get("region_code") or ""),
        "policy_version": str(metadata.get("policy_version") or metadata.get("source_version") or source.get("policy_version") or ""),
        "source_url": str(source.get("source_url") or ""),
        "audit_node": str(source.get("audit_node") or ""),
        "effective_start_time": metadata.get("effective_start_time") or source.get("effective_start_time"),
        "effective_end_time": metadata.get("effective_end_time") or source.get("effective_end_time"),
        "metadata": metadata,
    }


def rollback_to_document(target_document_id: str, change_note: str) -> dict:
    """回滚到指定文档版本并刷新知识库。"""
    if hasattr(workflow.audit_store, "rollback_document"):
        return workflow.audit_store.rollback_document(target_document_uid=target_document_id, change_note=change_note)
    target = find_rag_document(target_document_id)
    if not target:
        raise HTTPException(status_code=404, detail="Document not found")
    group_uid = target.version_group_uid or target.id
    for document in list(workflow.rag_engine.documents()):
        if (document.version_group_uid or document.id) == group_uid and document.status == "published":
            workflow.rag_engine.delete_document(document.id)
            workflow.rag_engine.index_document(replace(document, status="deprecated"))
    workflow.rag_engine.delete_document(target_document_id)
    workflow.rag_engine.index_document(replace(target, status="published", change_note=change_note))
    return {"document_id": target_document_id, "status": "published"}


def find_rag_document(document_id: str):
    """在 RAG 引擎中查找指定文档。"""
    return next((document for document in workflow.rag_engine.documents() if document.id == document_id), None)


def compare_text(left: str, right: str) -> tuple[list[str], list[str]]:
    """生成两段文本的统一差异。"""
    left_lines = [line.strip() for line in left.splitlines() if line.strip()]
    right_lines = [line.strip() for line in right.splitlines() if line.strip()]
    left_set = set(left_lines)
    right_set = set(right_lines)
    return sorted(right_set - left_set), sorted(left_set - right_set)


async def stage_upload_file(file: UploadFile, *, document_id: str) -> Path:
    """将上传文件保存到临时目录。"""
    return stage_upload_bytes(await file.read(), filename=file.filename or "upload.bin", document_id=document_id)


def stage_upload_bytes(data: bytes, *, filename: str, document_id: str) -> Path:
    """将文件字节写入上传暂存目录。"""
    settings.effective_upload_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(filename)
    path = settings.effective_upload_dir / f"{document_id}-{filename}"
    path.write_bytes(data)
    return path


def stable_upload_document_id(data: bytes) -> str:
    """根据文件名和内容生成稳定文档 ID。"""
    return f"doc-{hashlib.sha256(data).hexdigest()[:32]}"


def safe_filename(value: str) -> str:
    """清理上传文件名中的不安全字符。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "upload.bin"


def sse_event_stream(events: Iterable[dict]) -> Iterable[str]:
    """将文本片段转换为 SSE 数据流。"""
    for event in events:
        yield sse_event(str(event.get("event") or "event"), event.get("data") or {})


def sse_response(response: ChatResponse) -> Iterable[str]:
    """包装工作流事件为 SSE 响应。"""
    yield sse_event("run_started", {"session_id": response.session_id, "audit_id": response.audit_id})
    for event in response.events:
        yield sse_event(str(event.get("event") or "event"), event.get("data") or {})
    if response.sources:
        yield sse_event("sources", {"items": [source.model_dump() for source in response.sources]})
    for chunk in chunk_text(response.answer):
        yield sse_event("token", {"content": chunk})
    yield sse_event(
        "done",
        {
            "session_id": response.session_id,
            "audit_id": response.audit_id,
            "intent": response.intent,
            "route": response.route,
            "sub_intent": response.sub_intent,
            "multi_question": response.multi_question,
            "question_plan": response.question_plan,
            "sub_tasks": response.sub_tasks,
            "need_ticket_selection": response.need_ticket_selection,
            "ticket_selection_reason": response.ticket_selection_reason,
            "tickets": [ticket.model_dump() for ticket in response.tickets],
            "ticket_context": response.ticket_context,
            "retrieval_chain": response.retrieval_chain,
            "refused": response.refused,
            "refuse_reason": response.refuse_reason,
            "suggested_actions": [action.model_dump() for action in response.suggested_actions],
        },
    )


def sse_event(event: str, data: dict) -> str:
    """序列化单个 SSE 事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, *, size: int = 24) -> Iterable[str]:
    """按固定长度切分响应文本。"""
    value = str(text or "")
    if not value:
        return []
    return (value[index : index + size] for index in range(0, len(value), size))


app = create_app()
