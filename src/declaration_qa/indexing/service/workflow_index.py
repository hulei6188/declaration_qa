"""工作流索引写入入口。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument

if TYPE_CHECKING:
    from declaration_qa.application.workflow import DeclarationQAWorkflow


def index_document_for_workflow(
    workflow: DeclarationQAWorkflow,
    document: KnowledgeDocument,
    *,
    task_id: str = "",
    stage_callback=None,
) -> int:
    """将知识文档通过分阶段 ETL 写入工作流索引和持久化存储。

    文档先以 ``indexing`` 临时版本落库和写索引；普通问答默认只检索
    ``published``，因此临时版本不可见。全部索引写入成功后，再将同一
    document/chunk id 以目标状态重写一次，完成版本切换。
    """
    final_status = document.status or "published"
    final_revision = build_knowledge_revision(task_id)
    staging_revision = f"{final_revision}:staging"

    stage_callback = stage_callback or (lambda *args, **kwargs: None)
    staging_document = document_variant(
        document,
        status="indexing",
        revision=staging_revision,
        task_id=task_id,
        visible=False,
    )
    staging_chunks = chunks_for_storage(
        workflow.rag_engine.document_chunks(staging_document),
        document_status=staging_document.status,
    )
    stage_callback(
        "transform",
        35,
        "Document transformed into hierarchical chunks",
        {"chunks": len(staging_chunks), "knowledge_revision": staging_revision},
    )
    if hasattr(workflow.audit_store, "persist_document"):
        workflow.audit_store.persist_document(staging_document, staging_chunks)
    stage_callback(
        "embedding_load",
        65,
        "Temporary revision is being embedded and indexed",
        {"chunks": len(staging_chunks), "knowledge_revision": staging_revision},
    )
    workflow.rag_engine.index_chunks(staging_document, staging_chunks)

    final_document = document_variant(
        document,
        status=final_status,
        revision=final_revision,
        task_id=task_id,
        visible=final_status == "published",
    )
    final_chunks = chunks_for_storage(
        workflow.rag_engine.document_chunks(final_document),
        document_status=final_document.status,
    )
    stage_callback(
        "atomic_publish",
        90,
        "All chunks indexed; switching knowledge revision",
        {"chunks": len(final_chunks), "knowledge_revision": final_revision, "status": final_status},
    )
    if hasattr(workflow.audit_store, "persist_document"):
        workflow.audit_store.persist_document(final_document, final_chunks)
    chunks_indexed = workflow.rag_engine.promote_indexed_chunks(final_document, final_chunks)
    return chunks_indexed


def document_variant(
    document: KnowledgeDocument,
    *,
    status: str,
    revision: str,
    task_id: str,
    visible: bool,
) -> KnowledgeDocument:
    """返回携带 ETL 可见性元数据的文档副本。"""
    metadata = {
        **document.metadata,
        "doc_status": status,
        "knowledge_revision": revision,
        "etl_task_id": task_id,
        "revision_stage": "active" if visible else "staging",
        "revision_visible": visible,
    }
    return replace(document, status=status, metadata=metadata)


def chunks_for_storage(chunks: list[KnowledgeChunk], *, document_status: str) -> list[KnowledgeChunk]:
    """将业务文档状态映射为片段表允许的存储状态。"""
    storage_status = chunk_storage_status(document_status)
    return [replace(chunk, status=storage_status) for chunk in chunks]


def chunk_storage_status(document_status: str) -> str:
    """返回数据库约束允许的片段状态。"""
    if document_status == "published":
        return "published"
    if document_status == "deprecated":
        return "deprecated"
    return "draft"


def build_knowledge_revision(task_id: str) -> str:
    """构建紧凑且单调递增的知识修订标识。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    suffix = str(task_id or "manual")[:8]
    return f"v{timestamp}-{suffix}"
