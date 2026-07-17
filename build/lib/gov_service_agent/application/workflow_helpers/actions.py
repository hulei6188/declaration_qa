"""前端建议动作生成辅助函数。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.application.state import GovServiceAgentState
from gov_service_agent.rag.router import INTENT_REJECTION_REASON, INTENT_TICKET_STATUS


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
    apply_context = state.get("apply_context") or {}
    if state.get("refused"):
        return [{"type": "human_service", "label": "联系人工客服", "payload": {"appVersion": app_version}}]
    if state.get("intent") in {INTENT_TICKET_STATUS}:
        return [
            {
                "type": "open_apply_detail",
                "label": "查看申请单详情",
                "payload": {"applyId": apply_context.get("apply_id"), "appVersion": app_version},
            }
        ]
    if state.get("intent") in {INTENT_REJECTION_REASON}:
        reject_info = state.get("reject_info") or {}
        return [
            {
                "type": "open_material_reupload",
                "label": "重新上传补正材料",
                "payload": {
                    "applyId": apply_context.get("apply_id"),
                    "materialCode": reject_info.get("material_code"),
                    "appVersion": app_version,
                },
            }
        ]
    return [
        {
            "type": "open_guide",
            "label": "查看办事指南",
            "payload": {"bizType": state.get("biz_type"), "appVersion": app_version},
        }
    ]

