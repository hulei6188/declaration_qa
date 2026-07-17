"""知识文档删除处理。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gov_service_agent.application.workflow import GovServiceAgentWorkflow


def process_document_delete(payload: dict[str, Any], *, workflow: GovServiceAgentWorkflow) -> dict[str, Any]:
    """删除知识文档及其索引。"""
    document_id = str(payload.get("document_id") or "")
    if not document_id:
        raise ValueError("document_id is required")
    workflow.rag_engine.delete_document(document_id)
    deleted = workflow.audit_store.delete_document(document_id) if hasattr(workflow.audit_store, "delete_document") else True
    return {"document_id": document_id, "deleted": deleted, "message": "Document deleted"}

