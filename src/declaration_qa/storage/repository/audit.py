"""审计日志和反馈存储。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, bindparam, text

from declaration_qa.storage.repository.base import EngineBackedRepository
from declaration_qa.storage.session_payloads import query_image_attachments


class AuditRepository(EngineBackedRepository):
    """审计日志和反馈写入仓储。"""
    def append(self, payload: dict[str, Any]) -> str:
        """追加一条审计记录并返回记录 ID。"""
        audit_uid = str(payload.get("audit_id") or "")
        if not audit_uid:
            from uuid import uuid4

            audit_uid = uuid4().hex
        with self.engine.begin() as connection:
            session_pk = self._ensure_session(connection, payload)
            self._insert_message(
                connection,
                session_id=session_pk,
                role="user",
                content=str(payload.get("query") or ""),
                intent=str(payload.get("intent") or ""),
                refused=False,
                refuse_reason="",
                sources=[],
                suggested_actions=[],
                meta={
                    "query_image": payload.get("query_image") or {},
                    "attachments": query_image_attachments(payload.get("query_image")),
                },
            )
            self._insert_message(
                connection,
                session_id=session_pk,
                role="assistant",
                content=str(payload.get("answer") or ""),
                intent=str(payload.get("intent") or ""),
                refused=bool(payload.get("refused")),
                refuse_reason=str(payload.get("refuse_reason") or ""),
                sources=payload.get("retrieved_sources") or [],
                suggested_actions=payload.get("suggested_actions") or [],
                meta={
                    "retrieval_chain": payload.get("retrieval_chain") or "",
                    "business_snapshot": payload.get("business_snapshot") or {},
                    "evidence_ok": payload.get("evidence_ok"),
                    "evidence_decision": payload.get("evidence_decision") or {},
                    "rag_status": payload.get("rag_status") or {},
                    "short_term_context": payload.get("short_term_context") or {},
                    "long_term_context": payload.get("long_term_context") or {},
                    "langfuse_trace_id": payload.get("langfuse_trace_id") or "",
                },
            )
            stmt = text(
                """
                INSERT INTO declaration_qa.audit_logs (
                    audit_uid, session_id, user_id, query, normalized_query, intent,
                    retrieval_chain, region_code, biz_type, apply_id, business_snapshot,
                    business_status, reject_info,
                    material_mapping, retrieval_filter, retrieved_sources, rag_status,
                    evidence_ok, evidence_decision, refused, refuse_reason, final_prompt, answer,
                    events, langfuse_trace_id
                )
                VALUES (
                    :audit_uid, :session_id, :user_id, :query, :normalized_query, :intent,
                    :retrieval_chain, :region_code, :biz_type, :apply_id, :business_snapshot,
                    :business_status, :reject_info,
                    :material_mapping, :retrieval_filter, :retrieved_sources, :rag_status,
                    :evidence_ok, :evidence_decision, :refused, :refuse_reason, :final_prompt, :answer,
                    :events, :langfuse_trace_id
                )
                ON CONFLICT (audit_uid) DO UPDATE SET answer = EXCLUDED.answer
                RETURNING audit_uid
                """
            ).bindparams(
                bindparam("business_snapshot", type_=JSON),
                bindparam("business_status", type_=JSON),
                bindparam("reject_info", type_=JSON),
                bindparam("material_mapping", type_=JSON),
                bindparam("retrieval_filter", type_=JSON),
                bindparam("retrieved_sources", type_=JSON),
                bindparam("rag_status", type_=JSON),
                bindparam("evidence_decision", type_=JSON),
                bindparam("events", type_=JSON),
            )
            connection.execute(
                stmt,
                {
                    "audit_uid": audit_uid,
                    "session_id": session_pk,
                    "user_id": int(payload.get("user_id") or 0),
                    "query": str(payload.get("query") or ""),
                    "normalized_query": str(payload.get("normalized_query") or ""),
                    "intent": str(payload.get("intent") or ""),
                    "retrieval_chain": str(payload.get("retrieval_chain") or ""),
                    "region_code": str(payload.get("region_code") or ""),
                    "biz_type": payload.get("biz_type"),
                    "apply_id": str(payload.get("apply_id") or ""),
                    "business_snapshot": payload.get("business_snapshot") or {},
                    "business_status": payload.get("business_status") or {},
                    "reject_info": payload.get("reject_info") or {},
                    "material_mapping": payload.get("material_mapping") or {},
                    "retrieval_filter": payload.get("retrieval_filter") or {},
                    "retrieved_sources": payload.get("retrieved_sources") or [],
                    "rag_status": payload.get("rag_status") or {},
                    "evidence_ok": bool(payload.get("evidence_ok")),
                    "evidence_decision": payload.get("evidence_decision") or {},
                    "refused": bool(payload.get("refused")),
                    "refuse_reason": str(payload.get("refuse_reason") or ""),
                    "final_prompt": str(payload.get("final_prompt") or ""),
                    "answer": str(payload.get("answer") or ""),
                    "events": payload.get("events") or [],
                    "langfuse_trace_id": str(payload.get("langfuse_trace_id") or ""),
                },
            )
        return audit_uid

    def get(self, audit_id: str) -> dict[str, Any] | None:
        """按 ID 查询记录并返回副本。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM declaration_qa.audit_logs WHERE audit_uid = :audit_uid"),
                {"audit_uid": audit_id},
            ).mappings().first()
        return dict(row) if row else None

    def add_feedback(self, *, audit_id: str, rating: str, comment: str) -> dict[str, Any]:
        """保存用户对问答结果的反馈。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT id, user_id FROM declaration_qa.audit_logs WHERE audit_uid = :audit_uid"),
                {"audit_uid": audit_id},
            ).mappings().first()
            if not row:
                raise ValueError("Audit log not found")
            connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.feedback (audit_log_id, user_id, rating, comment)
                    VALUES (:audit_log_id, :user_id, :rating, :comment)
                    """
                ),
                {"audit_log_id": row["id"], "user_id": row["user_id"], "rating": rating, "comment": comment},
            )
        return {"audit_id": audit_id, "rating": rating, "comment": comment}

    def _ensure_session(self, connection, payload: dict[str, Any]) -> int:
        """确保会话记录存在并返回会话 ID。"""
        row = connection.execute(
            text(
                """
                INSERT INTO declaration_qa.sessions (session_uid, user_id, region_code, biz_type, apply_id)
                VALUES (:session_uid, :user_id, :region_code, :biz_type, :apply_id)
                ON CONFLICT (session_uid) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP,
                    biz_type = COALESCE(EXCLUDED.biz_type, declaration_qa.sessions.biz_type),
                    apply_id = COALESCE(NULLIF(EXCLUDED.apply_id, ''), declaration_qa.sessions.apply_id)
                RETURNING id
                """
            ),
            {
                "session_uid": str(payload.get("session_id") or ""),
                "user_id": int(payload.get("user_id") or 0),
                "region_code": str(payload.get("region_code") or "340000"),
                "biz_type": payload.get("biz_type"),
                "apply_id": str(payload.get("apply_id") or ""),
            },
        ).mappings().first()
        return int(row["id"])

    def _insert_message(
        self,
        connection,
        *,
        session_id: int,
        role: str,
        content: str,
        intent: str,
        refused: bool,
        refuse_reason: str,
        sources: list[dict[str, Any]],
        suggested_actions: list[dict[str, Any]],
        meta: dict[str, Any],
    ) -> None:
        """插入一条会话消息。"""
        stmt = text(
            """
            INSERT INTO declaration_qa.messages (
                session_id, role, content, intent, refused, refuse_reason,
                sources, suggested_actions, meta
            )
            VALUES (
                :session_id, :role, :content, :intent, :refused, :refuse_reason,
                :sources, :suggested_actions, :meta
            )
            """
        ).bindparams(
            bindparam("sources", type_=JSON),
            bindparam("suggested_actions", type_=JSON),
            bindparam("meta", type_=JSON),
        )
        connection.execute(
            stmt,
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "intent": intent,
                "refused": refused,
                "refuse_reason": refuse_reason,
                "sources": sources,
                "suggested_actions": suggested_actions,
                "meta": meta,
            },
        )

