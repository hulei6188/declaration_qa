"""知识库管理和索引任务路由。"""

from __future__ import annotations

from uuid import uuid4

from fastapi import Depends, File, Form, HTTPException, UploadFile

from gov_service_agent.api.auth import RequestContext, require_service_or_user
from gov_service_agent.api.schemas import (
    DeleteDocumentResponse,
    DocumentCompareResponse,
    DocumentRollbackRequest,
    DocumentRolloutRequest,
    DocumentRolloutResponse,
    DocumentStatusResponse,
    DocumentVersionRequest,
    DocumentVersionResponse,
    IndexJobResponse,
    KnowledgeBatchUploadResponse,
    KnowledgeDocumentAdminRecord,
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
    KnowledgeUploadResponse,
    ReindexResponse,
    RetryIndexJobResponse,
)
from gov_service_agent.api.server.dependencies import auth_context
from gov_service_agent.api.server.knowledge_admin import (
    compare_text,
    create_version,
    ensure_knowledge_base_fresh,
    get_document_record,
    list_document_records,
    rollback_to_document,
    set_document_status,
    update_rollout,
)
from gov_service_agent.api.server.runtime import index_job_store, index_queue, settings
from gov_service_agent.api.server.uploads import batch_upload_title, enqueue_upload_document, parse_metadata_form
from gov_service_agent.rag.engine.chunking import normalize_chunking_strategy


def register_knowledge_routes(app):
    """注册知识库管理和索引任务路由。"""
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
