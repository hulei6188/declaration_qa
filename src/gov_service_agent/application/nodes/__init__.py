"""GovServiceAgentWorkflow 使用的节点集合。"""

from __future__ import annotations

from gov_service_agent.application.nodes.agent_runtime import AgentRuntimeNodes
from gov_service_agent.application.nodes.preprocessing import PreprocessingNodes
from gov_service_agent.application.nodes.response import ResponseNodes
from gov_service_agent.application.nodes.retrieval import RetrievalNodes


class ApplicationNodes(
    PreprocessingNodes,
    AgentRuntimeNodes,
    RetrievalNodes,
    ResponseNodes,
):
    """GovServiceAgentWorkflow 使用的全部 LangGraph 节点方法。"""
