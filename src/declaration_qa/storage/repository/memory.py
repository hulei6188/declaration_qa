"""长期上下文记忆仓储。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, bindparam, text

from declaration_qa.memory.models import ContextMemoryItem
from declaration_qa.storage.repository.base import EngineBackedRepository


class ContextMemoryRepository(EngineBackedRepository):
    """PostgreSQL 长期上下文记忆仓储。"""

    def upsert_memory(self, memory: ContextMemoryItem) -> None:
        """写入或更新长期记忆。"""
        stmt = text(
            """
            INSERT INTO declaration_qa.context_memories (
                memory_uid, user_id, session_uid, memory_type, content, summary,
                source_audit_uid, source_query, source_answer,
                region_code, biz_type, importance, status, metadata
            )
            VALUES (
                :memory_uid, :user_id, :session_uid, :memory_type, :content, :summary,
                :source_audit_uid, :source_query, :source_answer,
                :region_code, :biz_type, :importance, :status, :metadata
            )
            ON CONFLICT (memory_uid) DO UPDATE SET
                content = EXCLUDED.content,
                summary = EXCLUDED.summary,
                source_audit_uid = EXCLUDED.source_audit_uid,
                source_query = EXCLUDED.source_query,
                source_answer = EXCLUDED.source_answer,
                region_code = EXCLUDED.region_code,
                biz_type = EXCLUDED.biz_type,
                importance = EXCLUDED.importance,
                status = EXCLUDED.status,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
            """
        ).bindparams(bindparam("metadata", type_=JSON))
        with self.engine.begin() as connection:
            connection.execute(
                stmt,
                {
                    "memory_uid": memory.id,
                    "user_id": memory.user_id,
                    "session_uid": memory.session_id,
                    "memory_type": memory.memory_type,
                    "content": memory.content,
                    "summary": memory.summary,
                    "source_audit_uid": memory.source_audit_id,
                    "source_query": memory.source_query,
                    "source_answer": memory.source_answer,
                    "region_code": memory.region_code,
                    "biz_type": memory.biz_type,
                    "importance": memory.importance,
                    "status": memory.status,
                    "metadata": memory.metadata,
                },
            )

    def get_memories_by_ids(self, memory_ids: list[str]) -> list[ContextMemoryItem]:
        """按记忆 ID 批量读取。"""
        ids = unique_ids(memory_ids)
        if not ids:
            return []
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT memory_uid, user_id, session_uid, memory_type, content, summary,
                           source_audit_uid, source_query, source_answer,
                           region_code, biz_type, importance, status, metadata, created_at
                    FROM declaration_qa.context_memories
                    WHERE memory_uid = ANY(:memory_ids)
                      AND status = 'published'
                    """
                ),
                {"memory_ids": ids},
            ).mappings().all()
        by_id = {str(row["memory_uid"]): memory_from_row(row) for row in rows}
        return [by_id[memory_id] for memory_id in ids if memory_id in by_id]

    def list_user_memories(self, *, user_id: int, limit: int = 100) -> list[ContextMemoryItem]:
        """读取指定用户最近长期记忆。"""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT memory_uid, user_id, session_uid, memory_type, content, summary,
                           source_audit_uid, source_query, source_answer,
                           region_code, biz_type, importance, status, metadata, created_at
                    FROM declaration_qa.context_memories
                    WHERE user_id = :user_id
                      AND status = 'published'
                    ORDER BY updated_at DESC, id DESC
                    LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": max(1, int(limit or 100))},
            ).mappings().all()
        return [memory_from_row(row) for row in rows]


def memory_from_row(row: Any) -> ContextMemoryItem:
    """将数据库行转换为长期记忆对象。"""
    payload = dict(row)
    return ContextMemoryItem(
        id=str(payload.get("memory_uid") or ""),
        user_id=int(payload.get("user_id") or 0),
        session_id=str(payload.get("session_uid") or ""),
        memory_type=str(payload.get("memory_type") or "qa_fact"),
        content=str(payload.get("content") or ""),
        summary=str(payload.get("summary") or ""),
        source_audit_id=str(payload.get("source_audit_uid") or ""),
        source_query=str(payload.get("source_query") or ""),
        source_answer=str(payload.get("source_answer") or ""),
        region_code=str(payload.get("region_code") or ""),
        biz_type=str(payload.get("biz_type") or ""),
        importance=float(payload.get("importance") or 0.5),
        status=str(payload.get("status") or "published"),
        metadata=dict(payload.get("metadata") or {}),
        created_at=str(payload.get("created_at") or ""),
    )


def unique_ids(values: list[Any]) -> list[str]:
    """保序去重。"""
    seen = set()
    output = []
    for value in values:
        text_value = str(value or "").strip()
        if not text_value or text_value in seen:
            continue
        seen.add(text_value)
        output.append(text_value)
    return output
