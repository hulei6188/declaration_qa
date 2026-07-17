"""与知识 RAG 完全隔离的工单查询工具。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from declaration_qa.application.tools.registry import ToolExecutionContext, ToolExecutionResult
from declaration_qa.core.business import BusinessDataAdapter


class TicketListQueryInput(BaseModel):
    """用户身份由执行上下文注入，模型无权指定。"""
    model_config = ConfigDict(extra="forbid")


class TicketDetailQueryInput(BaseModel):
    """使用可信事项编码批量查询申请详情。"""
    model_config = ConfigDict(extra="forbid")

    item_codes: list[Annotated[str, Field(min_length=1, max_length=160)]] = Field(
        min_length=1,
        max_length=50,
        description=(
            "事项编码数组；只能使用本轮可信业务结果中的真实 item_code，不得猜测、编造或使用"
            "历史对话中的 item_code；多个事项编码放入同一个数组"
        ),
    )
    detail_type: Literal["full", "status", "rejection"] = "full"


class TicketListQueryTool:
    """返回当前用户办理过的事项，由 Agent 自主确定 item_code。"""

    name = "ticket_list_query"
    description = (
        "查询当前登录用户办理过的政务事项列表，返回 item_code、事项名称和申请数量。"
    )
    input_model = TicketListQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        TicketListQueryInput.model_validate(arguments)
        items = self.adapter.list_ticket_items(
            user_id=context.user_id,
            region_code=context.region_code or None,
        )
        return ToolExecutionResult(
            tool_name=self.name,
            call_id=self.name,
            ok=bool(items),
            status="success" if items else "not_found",
            data={"items": items, "source_type": "business_tool"},
        )


class TicketDetailQueryTool:
    """按可信 user_id 和 Agent 选择的 item_codes 批量查询申请详情。"""

    name = "ticket_detail_query"
    description = (
        "按 item_codes 批量查询当前登录用户在多个事项下的全部申请。"
        "每条申请包含 application_id、当前状态以及按时间排序的完整 "
        "status_history。"
    )
    input_model = TicketDetailQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = TicketDetailQueryInput.model_validate(arguments)
        applications = self.adapter.get_applications_by_items(
            user_id=context.user_id,
            item_codes=request.item_codes,
        )
        applications = [filter_application_detail(item, request.detail_type) for item in applications]
        return ToolExecutionResult(
            tool_name=self.name,
            call_id=self.name,
            ok=bool(applications),
            status="success" if applications else "not_found",
            data={
                "item_codes": request.item_codes,
                "detail_type": request.detail_type,
                "applications": applications,
                "source_type": "business_tool",
            },
        )


def filter_application_detail(application: dict, detail_type: str) -> dict:
    """保留 Agent 判断申请所需信息，并按详情类型裁剪状态事件。"""
    output = dict(application)
    history = [dict(event) for event in (application.get("status_history") or []) if isinstance(event, dict)]
    if detail_type == "rejection":
        rejection_events = [
            event
            for event in history
            if event.get("reason")
            or event.get("reason_code")
            or str(event.get("status") or "").lower() in {"rejected", "returned", "correction"}
        ]
        output["status_history"] = history
        output["matching_status_events"] = rejection_events
    else:
        output["status_history"] = history
    return output
