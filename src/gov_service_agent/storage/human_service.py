"""人工服务工单的内存存储实现。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def handoff_context(audit: dict[str, Any]) -> dict[str, Any]:
    """从审计记录抽取坐席需要的上下文，避免携带完整原始消息。"""
    return {
        "query": str(audit.get("query") or ""),
        "normalized_query": str(audit.get("normalized_query") or ""),
        "intent": str(audit.get("intent") or ""),
        "sub_intent": str(audit.get("sub_intent") or ""),
        "route": str(audit.get("route") or ""),
        "route_decision": dict(audit.get("route_decision") or {}),
        "region_code": str(audit.get("region_code") or ""),
        "biz_type": audit.get("biz_type"),
        "business_snapshot": dict(audit.get("business_snapshot") or {}),
        "dialog_task": dict(audit.get("dialog_task") or {}),
        "retrieved_sources": list(audit.get("retrieved_sources") or [])[:12],
        "tool_results": list(audit.get("tool_results") or [])[-12:],
        "suggested_actions": list(audit.get("suggested_actions") or []),
        "answer": str(audit.get("answer") or ""),
        "refused": bool(audit.get("refused")),
        "refuse_reason": str(audit.get("refuse_reason") or ""),
    }


class InMemoryHumanServiceStore:
    """本地开发和测试用人工服务工单仓储。"""

    def __init__(self) -> None:
        self._tickets: dict[str, dict[str, Any]] = {}

    def create_ticket(self, *, audit: dict[str, Any], user_id: int, session_id: str, note: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        context = handoff_context(audit)
        ticket_id = f"hs-{uuid4().hex}"
        ticket = {
            "ticket_id": ticket_id,
            "audit_id": str(audit.get("id") or audit.get("audit_id") or ""),
            "session_id": session_id or str(audit.get("session_id") or ""),
            "user_id": int(user_id),
            "status": "open",
            "priority": "high" if audit.get("refused") else "normal",
            "assigned_to": None,
            "query": context["query"],
            "intent": context["intent"],
            "route": context["route"],
            "context": context,
            "messages": ([{"role": "user", "content": note, "created_at": now}] if note else []),
            "created_at": now,
            "updated_at": now,
        }
        self._tickets[ticket_id] = ticket
        return dict(ticket)

    def list_tickets(self, *, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        values = [item for item in self._tickets.values() if not status or item["status"] == status]
        return [dict(item) for item in sorted(values, key=lambda value: value["updated_at"], reverse=True)[:limit]]

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        item = self._tickets.get(ticket_id)
        return dict(item) if item else None

    def claim_ticket(self, *, ticket_id: str, agent_id: int) -> dict[str, Any]:
        ticket = self._tickets.get(ticket_id)
        if not ticket:
            raise ValueError("Human service ticket not found")
        ticket.update({"status": "claimed", "assigned_to": int(agent_id), "updated_at": datetime.now(timezone.utc).isoformat()})
        return dict(ticket)

    def update_status(self, *, ticket_id: str, status: str) -> dict[str, Any]:
        ticket = self._tickets.get(ticket_id)
        if not ticket:
            raise ValueError("Human service ticket not found")
        ticket.update({"status": status, "updated_at": datetime.now(timezone.utc).isoformat()})
        return dict(ticket)

    def add_message(self, *, ticket_id: str, role: str, content: str, author_id: int | None = None) -> dict[str, Any]:
        ticket = self._tickets.get(ticket_id)
        if not ticket:
            raise ValueError("Human service ticket not found")
        message = {"role": role, "content": content, "author_id": author_id, "created_at": datetime.now(timezone.utc).isoformat()}
        ticket["messages"].append(message)
        ticket["updated_at"] = message["created_at"]
        return dict(ticket)
