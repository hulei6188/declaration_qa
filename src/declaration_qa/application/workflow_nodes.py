"""Combined node methods for DeclarationQAWorkflow."""

from __future__ import annotations

from declaration_qa.application.workflow_planning_nodes import WorkflowPlanningNodes
from declaration_qa.application.workflow_response_nodes import WorkflowResponseNodes
from declaration_qa.application.workflow_retrieval_nodes import WorkflowRetrievalNodes
from declaration_qa.application.workflow_ticket_nodes import WorkflowTicketNodes


class WorkflowNodes(
    WorkflowPlanningNodes,
    WorkflowTicketNodes,
    WorkflowRetrievalNodes,
    WorkflowResponseNodes,
):
    """All LangGraph node methods used by DeclarationQAWorkflow."""
