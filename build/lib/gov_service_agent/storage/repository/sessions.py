"""会话历史存储。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from gov_service_agent.storage.repository.base import EngineBackedRepository
from gov_service_agent.storage.repository.normalizers import normalize_session_row
from gov_service_agent.storage.session_payloads import normalize_message_row


class SessionRepository(EngineBackedRepository):
    """会话摘要、详情和删除仓储。"""
    def list_sessions(self, *, user_id: int) -> list[dict[str, Any]]:
        """按用户查询历史会话摘要列表。"""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT s.session_uid AS session_id, s.user_id, s.region_code, s.biz_type,
                           s.apply_id, s.title, s.updated_at,
                           COUNT(m.id)::int AS message_count
                    FROM declaration_qa.sessions s
                    LEFT JOIN declaration_qa.messages m ON m.session_id = s.id
                    WHERE s.user_id = :user_id
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                    """
                ),
                {"user_id": user_id},
            ).mappings().all()
        return [normalize_session_row(row) for row in rows]

    def get_session(self, *, user_id: int, session_id: str) -> dict[str, Any] | None:
        """按用户和会话 ID 查询完整会话详情。"""
        with self.engine.begin() as connection:
            session = connection.execute(
                text(
                    """
                    SELECT session_uid AS session_id, user_id, region_code, biz_type,
                           apply_id, title, updated_at
                    FROM declaration_qa.sessions
                    WHERE session_uid = :session_id AND user_id = :user_id
                    """
                ),
                {"session_id": session_id, "user_id": user_id},
            ).mappings().first()
            if not session:
                return None
            messages = connection.execute(
                text(
                    """
                    SELECT role, content, intent, refused, refuse_reason, sources,
                           suggested_actions, meta, created_at
                    FROM declaration_qa.messages
                    WHERE session_id = (
                        SELECT id FROM declaration_qa.sessions
                        WHERE session_uid = :session_id AND user_id = :user_id
                    )
                    ORDER BY created_at ASC, id ASC
                    """
                ),
                {"session_id": session_id, "user_id": user_id},
            ).mappings().all()
        payload = normalize_session_row(session)
        payload["message_count"] = len(messages)
        payload["messages"] = [normalize_message_row(message) for message in messages]
        return payload

    def recent_messages(self, *, user_id: int, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """按用户和会话 ID 查询最近消息。"""
        with self.engine.begin() as connection:
            messages = connection.execute(
                text(
                    """
                    SELECT role, content, intent, refused, refuse_reason, sources,
                           suggested_actions, meta, created_at
                    FROM declaration_qa.messages
                    WHERE session_id = (
                        SELECT id FROM declaration_qa.sessions
                        WHERE session_uid = :session_id AND user_id = :user_id
                    )
                    ORDER BY created_at DESC, id DESC
                    LIMIT :limit
                    """
                ),
                {"session_id": session_id, "user_id": user_id, "limit": max(1, int(limit or 10))},
            ).mappings().all()
        return [normalize_message_row(message) for message in reversed(messages)]

    def delete_session(self, *, user_id: int, session_id: str) -> bool:
        """按用户删除指定历史会话。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    DELETE FROM declaration_qa.sessions
                    WHERE session_uid = :session_id AND user_id = :user_id
                    RETURNING id
                    """
                ),
                {"session_id": session_id, "user_id": user_id},
            ).mappings().first()
        return bool(row)

