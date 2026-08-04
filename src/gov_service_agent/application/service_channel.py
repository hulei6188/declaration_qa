"""线上申报与线下预约的办理渠道识别。"""

from __future__ import annotations

import re
from typing import Any


ONLINE_CHANNEL = "online"
OFFLINE_CHANNEL = "offline"

ONLINE_MARKERS = (
    "线上办理",
    "网上办理",
    "在线办理",
    "线上申报",
    "网上申报",
    "在线申报",
    "提交申报",
    "开始申报",
    "全程网办",
    "网办",
)
OFFLINE_MARKERS = (
    "线下办理",
    "窗口办理",
    "现场办理",
    "预约办理",
    "现场预约",
    "到大厅办理",
    "去大厅办理",
    "到窗口办理",
    "去窗口办理",
    "预约",
)
INFORMATION_MARKERS = (
    "怎么",
    "如何",
    "需要什么",
    "要什么材料",
    "哪些材料",
    "条件",
    "流程",
    "费用",
    "多久",
    "时限",
    "地址",
    "电话",
    "是否",
    "能否",
    "可以吗",
    "是什么",
)
TRANSACTION_PATTERN = re.compile(
    r"(?:我要|我想|帮我|替我|给我|为我|现在|直接|开始)(?:去|直接)?(?:办理|办|申请)"
)
CHANNEL_SWITCH_PATTERN = re.compile(r"(?:改成|改为|切换到|转成|转为|换成|走)(?:线上|网上|在线|线下|窗口|现场)")


def detect_service_channel(query: str, dialog_task: dict[str, Any] | None = None) -> str:
    """识别本轮明确指定的办理渠道，并让进行中的线上申报保持在线渠道。"""
    text = compact_text(query)
    has_online = any(marker in text for marker in ONLINE_MARKERS)
    has_offline = any(marker in text for marker in OFFLINE_MARKERS)
    if has_online != has_offline:
        return ONLINE_CHANNEL if has_online else OFFLINE_CHANNEL
    task = dialog_task if isinstance(dialog_task, dict) else {}
    if task.get("task_type") == "application_submission" and task.get("status") in {"collecting", "ready"}:
        return ONLINE_CHANNEL
    return ""


def requires_service_channel_clarification(query: str, channel: str) -> bool:
    """正式办理但未说明线上或线下时，要求先选择渠道。"""
    if channel:
        return False
    text = compact_text(query)
    if not TRANSACTION_PATTERN.search(text):
        return False
    return not any(marker in text for marker in INFORMATION_MARKERS)


def requests_channel_switch(query: str, channel: str) -> bool:
    """判断用户是否明确要求切换当前办理渠道，而非只咨询另一渠道。"""
    if channel not in {ONLINE_CHANNEL, OFFLINE_CHANNEL}:
        return False
    text = compact_text(query)
    if CHANNEL_SWITCH_PATTERN.search(text):
        return True
    if any(marker in text for marker in INFORMATION_MARKERS):
        return False
    return bool(TRANSACTION_PATTERN.search(text) or text in {*ONLINE_MARKERS, *OFFLINE_MARKERS})


def filter_tool_definitions_for_channel(definitions: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    """让模型只能看到当前渠道允许的办理写链路。"""
    blocked: set[str] = set()
    if channel == ONLINE_CHANNEL:
        blocked = {"appointment_hall_query", "appointment_slot_query", "appointment_book"}
    elif channel == OFFLINE_CHANNEL:
        blocked = {"application_submit"}
    return [item for item in definitions if tool_definition_name(item) not in blocked]


def service_channel_clarification_answer() -> str:
    """返回不依赖外部事实的渠道选择提示。"""
    return "请选择办理方式。确认后我会按对应流程继续。"


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def tool_definition_name(item: dict[str, Any]) -> str:
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    return str(item.get("name") or function.get("name") or "")
