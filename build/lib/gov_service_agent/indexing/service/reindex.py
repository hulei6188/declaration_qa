"""知识库重建索引处理。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gov_service_agent.application.workflow import GovServiceAgentWorkflow


def process_reindex(workflow: GovServiceAgentWorkflow) -> dict[str, Any]:
    """重建当前知识库索引。"""
    if hasattr(workflow.audit_store, "list_indexed_knowledge"):
        documents, chunks = workflow.audit_store.list_indexed_knowledge()
        chunks_indexed = workflow.rag_engine.rebuild_indexed_data(documents, chunks)
    else:
        documents = (
            workflow.audit_store.list_documents()
            if hasattr(workflow.audit_store, "list_documents")
            else workflow.rag_engine.documents()
        )
        chunks_indexed = workflow.rag_engine.rebuild(documents)
    return {
        "documents_indexed": len(documents),
        "chunks_indexed": chunks_indexed,
        "message": "Knowledge base reindexed",
    }

