"""知识检索工具；Agent 规划查询，服务端控制执行边界。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from declaration_qa.application.tools.registry import ToolExecutionContext, ToolExecutionResult
from declaration_qa.application.workflow_helpers.intent import mask_sensitive_text


class KnowledgeSubQuery(BaseModel):
    """Agent 针对一个原子知识问题规划的检索表达。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=2,
        max_length=300,
        description=(
            "完整、独立、无代词歧义的公开知识检索表达。只能根据用户原始问题、对话和上传图片生成，"
            "禁止包含工单工具返回的数据。"
        ),
    )
    strategy: Literal["keyword", "semantic", "hybrid"] = Field(
        default="hybrid",
        description=(
            "检索偏好：keyword 偏重精确名称和法规术语；semantic 偏重语义表达；"
            "hybrid 使用默认混合检索。执行器始终保留双通道降级能力。"
        ),
    )
    aspect: str = Field(
        default="general",
        min_length=1,
        max_length=50,
        description="该查询覆盖的原子问题，例如 materials、process、template。",
    )

    @field_validator("text", "aspect")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class KnowledgeSearchInput(BaseModel):
    """模型只规划公开知识查询，可信上下文由服务端注入。"""

    model_config = ConfigDict(extra="forbid")

    queries: list[KnowledgeSubQuery] = Field(
        min_length=1,
        max_length=6,
        description=(
            "根据用户问题自主规划的原子知识查询。简单问题只生成一路；复杂问题按材料、流程、条件等证据需求拆分。"
        ),
    )


class KnowledgeSearchTool:
    """调用现有混合 RAG 的受控工具包装器。"""

    name = "knowledge_search"
    description = (
        "查询公开办事指南、申请材料、流程、条件、费用、时限、政策依据和样表。"
        "不得用于查询个人工单状态、退回原因或申请历史。"
    )
    input_model = KnowledgeSearchInput

    def __init__(self, search: Callable[[KnowledgeSearchInput, ToolExecutionContext], dict[str, Any]]) -> None:
        self.search = search

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = KnowledgeSearchInput.model_validate(arguments)
        result = self.search(request, context)
        data = {
            **result,
            "requested_queries": [
                {**query.model_dump(), "text": mask_sensitive_text(query.text)}
                for query in request.queries
            ],
            "executed_query": context.original_query,
            "query_isolation": "agent_planned_before_business_observation",
            "source_type": "knowledge_rag",
        }
        return ToolExecutionResult(
            tool_name=self.name,
            call_id=self.name,
            ok=bool(data.get("sources")),
            status="success" if data.get("sources") else "not_found",
            data=data,
        )
