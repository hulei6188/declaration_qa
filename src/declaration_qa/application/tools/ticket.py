"""与知识 RAG 完全隔离的工单查询工具。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from declaration_qa.application.tools.registry import ToolExecutionContext, ToolExecutionResult
from declaration_qa.core.business import BusinessDataAdapter


class TicketListQueryInput(BaseModel):
    """用户身份由执行上下文注入，模型无权指定。"""
    model_config = ConfigDict(extra="forbid")


class TicketDetailQueryInput(BaseModel):
    """按事项查询多次申请，可进一步限定 application 主键。"""
    model_config = ConfigDict(extra="forbid")

    item_code: str = Field(min_length=1, max_length=160)
    application_id: int | None = Field(default=None, ge=1)
    detail_type: Literal["full", "status", "rejection"] = "full"


class TicketListQueryTool:
    """返回当前用户办理过的事项，由 Agent 自主确定 item_code。"""

    name = "ticket_list_query"
    description = (
        "查询当前登录用户办理过的政务事项列表，返回 item_code、事项名称和申请数量。"
        "用户询问个人申请进度、退回原因或历史申请且尚未确定 item_code 时调用。"
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
    """按可信 user_id 和 Agent 选择的 item_code 查询申请详情。"""

    name = "ticket_detail_query"
    description = (
        "按 item_code 查询当前登录用户在该事项下的全部申请；每条申请包含 application_id、当前状态"
        "以及按时间排序的完整 status_history。可用 application_id 进一步限定某次申请。"
    )
    input_model = TicketDetailQueryInput

    def __init__(self, adapter: BusinessDataAdapter) -> None:
        self.adapter = adapter

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        request = TicketDetailQueryInput.model_validate(arguments)
        applications = self.adapter.get_applications_by_item(
            user_id=context.user_id,
            item_code=request.item_code,
            application_id=request.application_id,
        )
        applications = [filter_application_detail(item, request.detail_type) for item in applications]
        return ToolExecutionResult(
            tool_name=self.name,
            call_id=self.name,
            ok=bool(applications),
            status="success" if applications else "not_found",
            data={
                "item_code": request.item_code,
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
