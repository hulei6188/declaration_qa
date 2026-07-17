"""知识文档管理和索引刷新辅助函数。"""

from __future__ import annotations

from dataclasses import replace
from threading import Lock

from fastapi import HTTPException

from gov_service_agent.api.schemas import DocumentRolloutRequest, DocumentVersionRequest
from gov_service_agent.api.server.runtime import workflow
from gov_service_agent.rag.retrieval import sanitize_metadata


_knowledge_sync_lock = Lock()
try:
    _knowledge_base_revision = (
        workflow.audit_store.knowledge_base_revision()
        if hasattr(workflow.audit_store, "knowledge_base_revision")
        else ""
    )
except Exception:
    _knowledge_base_revision = ""

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
        if hasattr(workflow.audit_store, "list_indexed_knowledge"):
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

