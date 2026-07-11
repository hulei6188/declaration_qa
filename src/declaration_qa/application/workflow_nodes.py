"""DeclarationQAWorkflow 的组合节点方法。"""

from __future__ import annotations

from declaration_qa.application.workflow_planning_nodes import WorkflowPlanningNodes
from declaration_qa.application.workflow_agent_nodes import WorkflowAgentNodes
from declaration_qa.application.workflow_response_nodes import WorkflowResponseNodes
from declaration_qa.application.workflow_retrieval_nodes import WorkflowRetrievalNodes
from declaration_qa.application.workflow_ticket_nodes import WorkflowTicketNodes


class WorkflowNodes(
    WorkflowPlanningNodes,
    WorkflowAgentNodes,
    WorkflowTicketNodes,
    WorkflowRetrievalNodes,
    WorkflowResponseNodes,
):
    """DeclarationQAWorkflow 使用的全部 LangGraph 节点方法。"""
