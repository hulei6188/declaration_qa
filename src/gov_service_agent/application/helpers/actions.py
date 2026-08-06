"""前端建议动作生成辅助函数。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.application.state import GovServiceAgentState


def material_tags(reject: dict[str, Any]) -> list[str]:
    """根据退回信息生成材料标签。"""
    material_code = str(reject.get("material_code") or "")
    reject_code = str(reject.get("reject_code") or "")
    tags = []
    if "PHOTO" in reject_code or "PHOTO" in material_code:
        tags.extend(["照片规范", "图片清晰度", "上传规范"])
    if material_code:
        tags.append(material_code)
    if reject_code:
        tags.append(reject_code)
    return tags


def suggested_actions(state: GovServiceAgentState) -> list[dict[str, Any]]:
    """根据状态生成后续建议动作。"""
    app_version = state.get("app_version") or ""
    item_code = str(state.get("agent_selected_item_code") or "")
    application_id = state.get("agent_selected_application_id")
    route_decision = state.get("route_decision") if isinstance(state.get("route_decision"), dict) else {}
    if (
        state.get("routing_clarification_required")
        and route_decision.get("clarification_kind") == "service_channel"
    ):
        item_name = str(route_decision.get("item_name") or "").strip()
        return [
            {
                "type": "select_service_channel",
                "label": "线上申报",
                "payload": {
                    "channel": "online",
                    "query": f"线上申报{item_name}" if item_name else "线上申报",
                    "description": "在线按步骤填写表单并上传申报材料",
                },
            },
            {
                "type": "select_service_channel",
                "label": "线下预约",
                "payload": {
                    "channel": "offline",
                    "query": f"线下预约{item_name}" if item_name else "线下预约",
                    "description": "查询服务大厅和号源后预约办理",
                },
            },
        ]
    if state.get("refused"):
        return [{"type": "human_service", "label": "联系人工客服", "payload": {"appVersion": app_version}}]
    has_detail = any(
        isinstance(result, dict)
        and result.get("tool_name") == "ticket_detail_query"
        and result.get("ok")
        for result in (state.get("tool_results") or [])
    )
    reject_info = selected_rejection_event(state) if has_detail else {}
    if reject_info:
        return [
            {
                "type": "open_material_reupload",
                "label": "重新上传补正材料",
                "payload": {
                    "itemCode": item_code,
                    "applicationId": application_id,
                    "materialCode": reject_info.get("material_code"),
                    "appVersion": app_version,
                },
            }
        ]
    if has_detail:
        return [
            {
                "type": "open_apply_detail",
                "label": "查看申请单详情",
                "payload": {"itemCode": item_code, "applicationId": application_id, "appVersion": app_version},
            }
        ]
    return [
        {
            "type": "open_guide",
            "label": "查看办事指南",
            "payload": {"bizType": state.get("biz_type"), "appVersion": app_version},
        }
    ]


def selected_rejection_event(state: GovServiceAgentState) -> dict[str, Any]:
    """从 Agent 选择的申请完整状态历史中读取最近一次退回事件。"""
    selected_id = int(state.get("agent_selected_application_id") or 0)
    for result in reversed(state.get("tool_results") or []):
        if not isinstance(result, dict) or result.get("tool_name") != "ticket_detail_query" or not result.get("ok"):
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        for application in data.get("applications") or []:
            if not isinstance(application, dict) or int(application.get("application_id") or 0) != selected_id:
                continue
            events = [event for event in application.get("status_history") or [] if isinstance(event, dict)]
            rejected = [event for event in events if event.get("reason") or event.get("reason_code")]
            return dict(rejected[-1]) if rejected else {}
    return {}

