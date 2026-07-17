"""GovServiceAgentWorkflow 的组合节点方法。"""

from __future__ import annotations

from gov_service_agent.application.workflow_planning_nodes import WorkflowPlanningNodes
from gov_service_agent.application.workflow_response_nodes import WorkflowResponseNodes
from gov_service_agent.application.workflow_retrieval_nodes import WorkflowRetrievalNodes
from gov_service_agent.application.workflow_ticket_nodes import WorkflowTicketNodes


class WorkflowNodes(
    WorkflowPlanningNodes,
    WorkflowTicketNodes,
    WorkflowRetrievalNodes,
    WorkflowResponseNodes,
):
    """GovServiceAgentWorkflow 使用的全部 LangGraph 节点方法。"""
