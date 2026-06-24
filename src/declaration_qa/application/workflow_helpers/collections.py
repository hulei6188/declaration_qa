"""工作流中的有序去重辅助函数。"""

from __future__ import annotations

from typing import Any


def unique_tickets(tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 ticket_id 去重办件摘要。"""
    seen = set()
    output = []
    for ticket in tickets:
        ticket_id = str(ticket.get("ticket_id") or "")
        if not ticket_id or ticket_id in seen:
            continue
        seen.add(ticket_id)
        output.append(ticket)
    return output


def unique_strings(values: list[str]) -> list[str]:
    """按出现顺序去重字符串。"""
    seen = set()
    output = []
    for value in values:
        item = str(value or "")
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output

