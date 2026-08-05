"""申报与投诉表单的确定性状态迁移。"""

from __future__ import annotations

from typing import Any


TERMINAL_DIALOG_STATUSES = frozenset({"completed", "cancelled"})
ALLOWED_DIALOG_TRANSITIONS: dict[str, frozenset[str]] = {
    "": frozenset({"collecting", "ready"}),
    # 工具在确认预览中计算 ready，但中断前不会覆盖持久化 collecting 状态；
    # 人工确认后的成功写入因此允许 collecting 直接进入 completed。
    "collecting": frozenset({"collecting", "ready", "completed", "cancelled"}),
    "ready": frozenset({"ready", "collecting", "completed", "cancelled"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}
TOOL_TASK_TYPES = {
    "application_submit": "application_submission",
    "complaint_create": "complaint_create",
}


def transition_dialog_task(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    tool_name: str,
) -> tuple[dict[str, Any], str]:
    """校验工具返回的表单状态，拒绝跨任务或非法跳转。"""
    previous = dict(current or {})
    proposed = dict(candidate or {})
    if not proposed:
        return previous, ""
    expected_task_type = TOOL_TASK_TYPES.get(str(tool_name or ""))
    if not expected_task_type:
        return previous, "dialog_task_from_non_workflow_tool"
    if str(proposed.get("task_type") or "") != expected_task_type:
        return previous, "dialog_task_type_mismatch"
    previous_status = str(previous.get("status") or "")
    next_status = str(proposed.get("status") or "")
    if next_status not in ALLOWED_DIALOG_TRANSITIONS.get(previous_status, frozenset()):
        return previous, f"illegal_dialog_transition:{previous_status or 'empty'}->{next_status or 'empty'}"
    if previous and previous_status not in TERMINAL_DIALOG_STATUSES:
        if str(previous.get("task_type") or "") != str(proposed.get("task_type") or ""):
            return previous, "active_dialog_task_type_changed"
        previous_item = str(previous.get("item_code") or "")
        proposed_item = str(proposed.get("item_code") or "")
        if previous_item and proposed_item and previous_item != proposed_item:
            return previous, "active_dialog_item_changed"
    return proposed, ""


def cancel_active_dialog_task(task: dict[str, Any] | None) -> dict[str, Any]:
    """显式取消当前 collecting/ready 表单。"""
    current = dict(task or {})
    if str(current.get("status") or "") not in {"collecting", "ready"}:
        return current
    current["status"] = "cancelled"
    current["next_field"] = {}
    return current
