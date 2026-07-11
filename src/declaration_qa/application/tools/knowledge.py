"""知识检索工具；查询与过滤输入在工单工具执行前冻结。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from declaration_qa.application.tools.registry import ToolExecutionContext, ToolExecutionResult


class KnowledgeSearchInput(BaseModel):
    """模型可以说明检索焦点，但不能传入业务字段或过滤条件。"""
    model_config = ConfigDict(extra="forbid")

    query_focus: str = Field(default="", max_length=1000)


class KnowledgeSearchTool:
    """调用现有混合 RAG 的受控工具包装器。"""

    name = "knowledge_search"
    description = (
        "查询公开办事指南、申请材料、流程、条件、费用、时限、政策依据和样表。"
        "不得用于查询个人工单状态、退回原因或申请历史。"
    )
    input_model = KnowledgeSearchInput

    def __init__(self, search: Callable[[ToolExecutionContext], dict[str, Any]]) -> None:
        self.search = search

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = KnowledgeSearchInput.model_validate(arguments)
        result = self.search(context)
        data = {
            **result,
            "requested_focus": request.query_focus,
            "executed_query": context.original_query,
            "query_isolation": "frozen_before_business_tools",
            "source_type": "knowledge_rag",
        }
        return ToolExecutionResult(
            tool_name=self.name,
            call_id=self.name,
            ok=bool(data.get("sources")),
            status="success" if data.get("sources") else "not_found",
            data=data,
        )
