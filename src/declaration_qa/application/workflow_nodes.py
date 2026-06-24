"""Combined node mixin for DeclarationQAWorkflow."""

from __future__ import annotations

from declaration_qa.application.workflow_planning_nodes import WorkflowPlanningMixin
from declaration_qa.application.workflow_response_nodes import WorkflowResponseMixin
from declaration_qa.application.workflow_retrieval_nodes import WorkflowRetrievalMixin
from declaration_qa.application.workflow_ticket_nodes import WorkflowTicketMixin


class WorkflowNodeMixin(
    WorkflowPlanningMixin,
    WorkflowTicketMixin,
    WorkflowRetrievalMixin,
    WorkflowResponseMixin,
):
    """All LangGraph node methods used by DeclarationQAWorkflow."""
