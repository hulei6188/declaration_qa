"""工作流索引写入入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from declaration_qa.core.domain import KnowledgeDocument

if TYPE_CHECKING:
    from declaration_qa.application.workflow import DeclarationQAWorkflow


def index_document_for_workflow(workflow: DeclarationQAWorkflow, document: KnowledgeDocument) -> int:
    """将知识文档写入工作流索引和持久化存储。"""
    chunks = workflow.rag_engine.document_chunks(document)
    if hasattr(workflow.audit_store, "persist_document"):
        workflow.audit_store.persist_document(document, chunks)
    chunks_indexed = workflow.rag_engine.index_chunks(document, chunks)
    return chunks_indexed

