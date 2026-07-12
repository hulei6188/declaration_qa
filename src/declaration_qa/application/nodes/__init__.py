"""DeclarationQAWorkflow 使用的节点集合。"""

from __future__ import annotations

from declaration_qa.application.nodes.agent_runtime import AgentRuntimeNodes
from declaration_qa.application.nodes.preprocessing import PreprocessingNodes
from declaration_qa.application.nodes.response import ResponseNodes
from declaration_qa.application.nodes.retrieval import RetrievalNodes


class ApplicationNodes(
    PreprocessingNodes,
    AgentRuntimeNodes,
    RetrievalNodes,
    ResponseNodes,
):
    """DeclarationQAWorkflow 使用的全部 LangGraph 节点方法。"""
