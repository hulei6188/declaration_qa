"""工作流事件辅助函数。"""

from __future__ import annotations

from typing import Any


def add_event(state: dict[str, Any], event: str, data: dict[str, Any]) -> None:
    """向工作流事件列表追加事件。"""
    events = list(state.get("events") or [])
    events.append({"event": event, "data": data})
    state["events"] = events

