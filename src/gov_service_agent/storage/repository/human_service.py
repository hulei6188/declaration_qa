"""人工服务工单 PostgreSQL 仓储。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, bindparam, text

from gov_service_agent.storage.human_service import handoff_context
from gov_service_agent.storage.repository.base import EngineBackedRepository


class HumanServiceRepository(EngineBackedRepository):
    """保存转人工工单、坐席上下文和消息。"""

    def create_ticket(self, *, audit: dict[str, Any], user_id: int, session_id: str, note: str) -> dict[str, Any]:
        ticket_uid = f"hs-{uuid4().hex}"
        context = handoff_context(audit)
        with self.engine.begin() as connection:
            row = connection.execute(
                text("""
                    INSERT INTO declaration_qa.human_service_tickets
                        (ticket_uid, audit_uid, session_id, user_id, status, priority, query, intent, route, context)
                    VALUES (:ticket_uid, :audit_uid, :session_id, :user_id, 'open', :priority, :query, :intent, :route, :context)
                    RETURNING ticket_uid AS ticket_id, audit_uid, session_id, user_id, status, priority,
                              assigned_to, query, intent, route, context, created_at, updated_at
                """).bindparams(bindparam("context", type_=JSON)),
                {"ticket_uid": ticket_uid, "audit_uid": str(audit.get("id") or audit.get("audit_id") or ""), "session_id": session_id, "user_id": user_id, "priority": "high" if audit.get("refused") else "normal", "query": context["query"], "intent": context["intent"], "route": context["route"], "context": context},
            ).mappings().first()
            if note:
                connection.execute(text("INSERT INTO declaration_qa.human_service_messages (ticket_id, role, content) SELECT id, 'user', :content FROM declaration_qa.human_service_tickets WHERE ticket_uid = :ticket_uid"), {"ticket_uid": ticket_uid, "content": note})
        return self._with_messages(dict(row), ticket_uid)

    def list_tickets(self, *, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT ticket_uid AS ticket_id, audit_uid, session_id, user_id, status, priority, assigned_to, query, intent, route, context, created_at, updated_at FROM declaration_qa.human_service_tickets WHERE (:status = '' OR status = :status) ORDER BY updated_at DESC LIMIT :limit"), {"status": status, "limit": limit}).mappings().all()
        return [self._with_messages(dict(row), str(row["ticket_id"])) for row in rows]

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT ticket_uid AS ticket_id, audit_uid, session_id, user_id, status, priority, assigned_to, query, intent, route, context, created_at, updated_at FROM declaration_qa.human_service_tickets WHERE ticket_uid = :ticket_uid"), {"ticket_uid": ticket_id}).mappings().first()
        return self._with_messages(dict(row), ticket_id) if row else None

    def claim_ticket(self, *, ticket_id: str, agent_id: int) -> dict[str, Any]:
        row = self._update_ticket(ticket_id, "status = 'claimed', assigned_to = :agent_id", {"agent_id": agent_id})
        return row

    def update_status(self, *, ticket_id: str, status: str) -> dict[str, Any]:
        return self._update_ticket(ticket_id, "status = :status", {"status": status})

    def add_message(self, *, ticket_id: str, role: str, content: str, author_id: int | None = None) -> dict[str, Any]:
        with self.engine.begin() as connection:
            exists = connection.execute(text("SELECT id FROM declaration_qa.human_service_tickets WHERE ticket_uid = :ticket_uid"), {"ticket_uid": ticket_id}).scalar_one_or_none()
            if not exists:
                raise ValueError("Human service ticket not found")
            connection.execute(text("INSERT INTO declaration_qa.human_service_messages (ticket_id, role, content, author_id) VALUES (:ticket_id, :role, :content, :author_id)"), {"ticket_id": exists, "role": role, "content": content, "author_id": author_id})
        return self.get_ticket(ticket_id) or {}

    def _update_ticket(self, ticket_id: str, assignments: str, params: dict[str, Any]) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(text(f"UPDATE declaration_qa.human_service_tickets SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE ticket_uid = :ticket_uid RETURNING ticket_uid AS ticket_id, audit_uid, session_id, user_id, status, priority, assigned_to, query, intent, route, context, created_at, updated_at"), {"ticket_uid": ticket_id, **params}).mappings().first()
        if not row:
            raise ValueError("Human service ticket not found")
        return self._with_messages(dict(row), ticket_id)

    def _with_messages(self, ticket: dict[str, Any], ticket_id: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT role, content, author_id, created_at FROM declaration_qa.human_service_messages m JOIN declaration_qa.human_service_tickets t ON t.id = m.ticket_id WHERE t.ticket_uid = :ticket_uid ORDER BY m.created_at ASC, m.id ASC"), {"ticket_uid": ticket_id}).mappings().all()
        ticket["messages"] = [dict(row) for row in rows]
        return ticket
