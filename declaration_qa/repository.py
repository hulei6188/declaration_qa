from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, bindparam, create_engine, text
from sqlalchemy.engine import Engine

from declaration_qa.domain import KnowledgeChunk, KnowledgeDocument
from declaration_qa.source_types import canonicalize_source_type


class PostgresDeclarationQAStore:
    def __init__(self, database_url: str) -> None:
        self.engine: Engine = create_engine(database_url, future=True, pool_pre_ping=True)

    def init_schema(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "sql" / "postgresql" / "003_declaration_qa_schema.sql"
        with self.engine.begin() as connection:
            connection.exec_driver_sql(schema_path.read_text(encoding="utf-8"))

    def append(self, payload: dict[str, Any]) -> str:
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
                meta={},
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
                meta={"evidence_ok": payload.get("evidence_ok"), "rag_status": payload.get("rag_status") or {}},
            )
            stmt = text(
                """
                INSERT INTO declaration_qa.audit_logs (
                    audit_uid, session_id, user_id, query, normalized_query, intent,
                    region_code, biz_type, apply_id, business_status, reject_info,
                    material_mapping, retrieval_filter, retrieved_sources, rag_status,
                    evidence_ok, refused, refuse_reason, final_prompt, answer, events
                )
                VALUES (
                    :audit_uid, :session_id, :user_id, :query, :normalized_query, :intent,
                    :region_code, :biz_type, :apply_id, :business_status, :reject_info,
                    :material_mapping, :retrieval_filter, :retrieved_sources, :rag_status,
                    :evidence_ok, :refused, :refuse_reason, :final_prompt, :answer, :events
                )
                ON CONFLICT (audit_uid) DO UPDATE SET answer = EXCLUDED.answer
                RETURNING audit_uid
                """
            ).bindparams(
                bindparam("business_status", type_=JSON),
                bindparam("reject_info", type_=JSON),
                bindparam("material_mapping", type_=JSON),
                bindparam("retrieval_filter", type_=JSON),
                bindparam("retrieved_sources", type_=JSON),
                bindparam("rag_status", type_=JSON),
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
                    "region_code": str(payload.get("region_code") or ""),
                    "biz_type": payload.get("biz_type"),
                    "apply_id": str(payload.get("apply_id") or ""),
                    "business_status": payload.get("business_status") or {},
                    "reject_info": payload.get("reject_info") or {},
                    "material_mapping": payload.get("material_mapping") or {},
                    "retrieval_filter": payload.get("retrieval_filter") or {},
                    "retrieved_sources": payload.get("retrieved_sources") or [],
                    "rag_status": payload.get("rag_status") or {},
                    "evidence_ok": bool(payload.get("evidence_ok")),
                    "refused": bool(payload.get("refused")),
                    "refuse_reason": str(payload.get("refuse_reason") or ""),
                    "final_prompt": str(payload.get("final_prompt") or ""),
                    "answer": str(payload.get("answer") or ""),
                    "events": payload.get("events") or [],
                },
            )
        return audit_uid

    def get(self, audit_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM declaration_qa.audit_logs WHERE audit_uid = :audit_uid"),
                {"audit_uid": audit_id},
            ).mappings().first()
        return dict(row) if row else None

    def get_apply_context(self, *, user_id: int, apply_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        a.apply_uid AS apply_id,
                        a.user_id,
                        a.biz_type,
                        a.region_code,
                        COALESCE(s.status, a.status) AS status,
                        COALESCE(s.current_node, a.current_node) AS current_node,
                        COALESCE(s.node_name, a.node_name) AS node_name,
                        r.reject_code,
                        r.reject_reason,
                        r.material_code,
                        r.material_name,
                        r.audit_node
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    LEFT JOIN declaration_qa.business_reject_infos r
                        ON r.apply_id = a.id AND r.is_current = TRUE
                    WHERE a.apply_uid = :apply_uid
                      AND a.user_id = :user_id
                      AND a.deleted_at IS NULL
                    """
                ),
                {"apply_uid": str(apply_id or ""), "user_id": int(user_id)},
            ).mappings().first()
        return normalize_business_row(row) if row else None

    def verify_apply_access(self, *, user_id: int, apply_id: str) -> bool:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT 1
                    FROM declaration_qa.business_applications
                    WHERE apply_uid = :apply_uid
                      AND user_id = :user_id
                      AND deleted_at IS NULL
                    """
                ),
                {"apply_uid": str(apply_id or ""), "user_id": int(user_id)},
            ).first()
        return row is not None

    def get_status(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        apply_id = str(apply_context.get("apply_id") or "").strip()
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        a.apply_uid AS apply_id,
                        a.biz_type,
                        a.region_code,
                        COALESCE(s.status, a.status) AS status,
                        COALESCE(s.current_node, a.current_node) AS current_node,
                        COALESCE(s.node_name, a.node_name) AS node_name
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_application_statuses s
                        ON s.apply_id = a.id AND s.is_current = TRUE
                    WHERE a.apply_uid = :apply_uid
                      AND a.deleted_at IS NULL
                    """
                ),
                {"apply_uid": apply_id},
            ).mappings().first()
        merged = {**apply_context, **(normalize_business_row(row) if row else {})}
        return {
            "apply_id": merged.get("apply_id"),
            "biz_type": merged.get("biz_type"),
            "region_code": merged.get("region_code"),
            "status": merged.get("status"),
            "current_node": merged.get("current_node"),
            "node_name": merged.get("node_name"),
        }

    def get_reject_info(self, apply_context: dict[str, Any]) -> dict[str, Any]:
        apply_id = str(apply_context.get("apply_id") or "").strip()
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT
                        a.apply_uid AS apply_id,
                        a.biz_type,
                        a.region_code,
                        r.reject_code,
                        r.reject_reason,
                        r.material_code,
                        r.material_name,
                        r.audit_node
                    FROM declaration_qa.business_applications a
                    LEFT JOIN declaration_qa.business_reject_infos r
                        ON r.apply_id = a.id AND r.is_current = TRUE
                    WHERE a.apply_uid = :apply_uid
                      AND a.deleted_at IS NULL
                    """
                ),
                {"apply_uid": apply_id},
            ).mappings().first()
        merged = {**apply_context, **(normalize_business_row(row) if row else {})}
        return {
            "apply_id": merged.get("apply_id"),
            "biz_type": merged.get("biz_type"),
            "region_code": merged.get("region_code"),
            "reject_code": merged.get("reject_code"),
            "reject_reason": merged.get("reject_reason"),
            "material_code": merged.get("material_code"),
            "material_name": merged.get("material_name"),
            "audit_node": merged.get("audit_node"),
        }

    def add_feedback(self, *, audit_id: str, rating: str, comment: str) -> dict[str, Any]:
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

    def create_eval_dataset(self, *, name: str, description: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
        dataset_uid = f"eval-ds-{uuid4().hex}"
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.eval_datasets (dataset_uid, name, description)
                    VALUES (:dataset_uid, :name, :description)
                    RETURNING dataset_uid AS dataset_id, name, description, created_at
                    """
                ),
                {"dataset_uid": dataset_uid, "name": name, "description": description},
            ).mappings().first()
        for case in cases:
            self.add_eval_case(dataset_id=dataset_uid, case=case)
        return {**normalize_eval_dataset(row), "case_count": len(self.list_eval_cases(dataset_uid))}

    def list_eval_datasets(self) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT d.dataset_uid AS dataset_id, d.name, d.description, d.created_at,
                           COUNT(c.id)::int AS case_count
                    FROM declaration_qa.eval_datasets d
                    LEFT JOIN declaration_qa.eval_cases c ON c.dataset_id = d.id
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                    """
                )
            ).mappings().all()
        return [normalize_eval_dataset(row) for row in rows]

    def add_eval_case(self, *, dataset_id: str, case: dict[str, Any]) -> dict[str, Any]:
        case_uid = f"eval-case-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_cases (
                case_uid, dataset_id, query, user_id, region_code, policy_version,
                biz_type, apply_id, apply_context, expected_intent, expected_refused,
                expected_doc_ids, expected_answer_keywords, tags
            )
            SELECT :case_uid, id, :query, :user_id, :region_code, :policy_version,
                   :biz_type, :apply_id, :apply_context, :expected_intent, :expected_refused,
                   :expected_doc_ids, :expected_answer_keywords, :tags
            FROM declaration_qa.eval_datasets
            WHERE dataset_uid = :dataset_uid
            RETURNING case_uid AS case_id, :dataset_uid AS dataset_id, query, user_id,
                      region_code, policy_version, biz_type, apply_id, apply_context,
                      expected_intent, expected_refused, expected_doc_ids,
                      expected_answer_keywords, tags, created_at
            """
        ).bindparams(
            bindparam("apply_context", type_=JSON),
            bindparam("expected_doc_ids", type_=JSON),
            bindparam("expected_answer_keywords", type_=JSON),
            bindparam("tags", type_=JSON),
        )
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
                {
                    "case_uid": case_uid,
                    "dataset_uid": dataset_id,
                    "query": str(case.get("query") or ""),
                    "user_id": int(case.get("user_id") or 1),
                    "region_code": str(case.get("region_code") or "340000"),
                    "policy_version": str(case.get("policy_version") or "2025"),
                    "biz_type": case.get("biz_type"),
                    "apply_id": case.get("apply_id"),
                    "apply_context": case.get("apply_context"),
                    "expected_intent": str(case.get("expected_intent") or ""),
                    "expected_refused": case.get("expected_refused"),
                    "expected_doc_ids": list(case.get("expected_doc_ids") or []),
                    "expected_answer_keywords": list(case.get("expected_answer_keywords") or []),
                    "tags": list(case.get("tags") or []),
                },
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation dataset not found")
        return normalize_eval_case(row)

    def list_eval_cases(self, dataset_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT c.case_uid AS case_id, d.dataset_uid AS dataset_id, c.query,
                           c.user_id, c.region_code, c.policy_version, c.biz_type,
                           c.apply_id, c.apply_context, c.expected_intent,
                           c.expected_refused, c.expected_doc_ids,
                           c.expected_answer_keywords, c.tags, c.created_at
                    FROM declaration_qa.eval_cases c
                    JOIN declaration_qa.eval_datasets d ON d.id = c.dataset_id
                    WHERE d.dataset_uid = :dataset_uid
                    ORDER BY c.created_at ASC, c.id ASC
                    """
                ),
                {"dataset_uid": dataset_id},
            ).mappings().all()
        return [normalize_eval_case(row) for row in rows]

    def create_eval_run(self, *, dataset_id: str, note: str, top_k: int) -> dict[str, Any]:
        run_uid = f"eval-run-{uuid4().hex}"
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.eval_runs (run_uid, dataset_id, status, note, top_k)
                    SELECT :run_uid, id, 'running', :note, :top_k
                    FROM declaration_qa.eval_datasets
                    WHERE dataset_uid = :dataset_uid
                    RETURNING run_uid AS run_id, :dataset_uid AS dataset_id, status, note,
                              top_k, metrics, error_message, created_at, completed_at
                    """
                ),
                {"run_uid": run_uid, "dataset_uid": dataset_id, "note": note, "top_k": int(top_k)},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation dataset not found")
        return normalize_eval_run(row, items=[])

    def append_eval_run_item(self, *, run_id: str, item: dict[str, Any]) -> dict[str, Any]:
        item_uid = f"eval-item-{uuid4().hex}"
        stmt = text(
            """
            INSERT INTO declaration_qa.eval_run_items (
                item_uid, run_id, case_id, query, expected_intent, actual_intent,
                expected_refused, actual_refused, expected_doc_ids, retrieved_doc_ids,
                expected_answer_keywords, answer, audit_uid, recall_hit, refusal_match,
                intent_match, keyword_hit
            )
            SELECT :item_uid, r.id, c.id, :query, :expected_intent, :actual_intent,
                   :expected_refused, :actual_refused, :expected_doc_ids, :retrieved_doc_ids,
                   :expected_answer_keywords, :answer, :audit_uid, :recall_hit,
                   :refusal_match, :intent_match, :keyword_hit
            FROM declaration_qa.eval_runs r
            LEFT JOIN declaration_qa.eval_cases c ON c.case_uid = :case_uid
            WHERE r.run_uid = :run_uid
            RETURNING item_uid AS item_id, :case_uid AS case_id, query, expected_intent,
                      actual_intent, expected_refused, actual_refused, expected_doc_ids,
                      retrieved_doc_ids, expected_answer_keywords, answer, audit_uid,
                      recall_hit, refusal_match, intent_match, keyword_hit,
                      annotation_label, annotation_comment
            """
        ).bindparams(
            bindparam("expected_doc_ids", type_=JSON),
            bindparam("retrieved_doc_ids", type_=JSON),
            bindparam("expected_answer_keywords", type_=JSON),
        )
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
                {
                    "item_uid": item_uid,
                    "run_uid": run_id,
                    "case_uid": str(item.get("case_id") or ""),
                    "query": str(item.get("query") or ""),
                    "expected_intent": str(item.get("expected_intent") or ""),
                    "actual_intent": str(item.get("actual_intent") or ""),
                    "expected_refused": item.get("expected_refused"),
                    "actual_refused": bool(item.get("actual_refused")),
                    "expected_doc_ids": list(item.get("expected_doc_ids") or []),
                    "retrieved_doc_ids": list(item.get("retrieved_doc_ids") or []),
                    "expected_answer_keywords": list(item.get("expected_answer_keywords") or []),
                    "answer": str(item.get("answer") or ""),
                    "audit_uid": str(item.get("audit_id") or ""),
                    "recall_hit": item.get("recall_hit"),
                    "refusal_match": item.get("refusal_match"),
                    "intent_match": item.get("intent_match"),
                    "keyword_hit": item.get("keyword_hit"),
                },
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation run not found")
        return normalize_eval_run_item(row)

    def finish_eval_run(self, *, run_id: str, status: str, metrics: dict[str, Any], error_message: str = "") -> dict[str, Any]:
        stmt = text(
            """
            UPDATE declaration_qa.eval_runs SET
                status = :status,
                metrics = :metrics,
                error_message = :error_message,
                completed_at = CURRENT_TIMESTAMP
            WHERE run_uid = :run_uid
            RETURNING run_uid AS run_id, status, metrics, error_message, completed_at
            """
        ).bindparams(bindparam("metrics", type_=JSON))
        with self.engine.begin() as connection:
            row = connection.execute(
                stmt,
                {"run_uid": run_id, "status": status, "metrics": metrics, "error_message": error_message},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation run not found")
        return self.get_eval_run(run_id) or normalize_eval_run(row, items=[])

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            run = connection.execute(
                text(
                    """
                    SELECT r.run_uid AS run_id, d.dataset_uid AS dataset_id, r.status,
                           r.note, r.top_k, r.metrics, r.error_message,
                           r.created_at, r.completed_at
                    FROM declaration_qa.eval_runs r
                    JOIN declaration_qa.eval_datasets d ON d.id = r.dataset_id
                    WHERE r.run_uid = :run_uid
                    """
                ),
                {"run_uid": run_id},
            ).mappings().first()
            if not run:
                return None
            items = connection.execute(
                text(
                    """
                    SELECT i.item_uid AS item_id, c.case_uid AS case_id, i.query,
                           i.expected_intent, i.actual_intent, i.expected_refused,
                           i.actual_refused, i.expected_doc_ids, i.retrieved_doc_ids,
                           i.expected_answer_keywords, i.answer, i.audit_uid,
                           i.recall_hit, i.refusal_match, i.intent_match, i.keyword_hit,
                           i.annotation_label, i.annotation_comment
                    FROM declaration_qa.eval_run_items i
                    LEFT JOIN declaration_qa.eval_cases c ON c.id = i.case_id
                    WHERE i.run_id = (
                        SELECT id FROM declaration_qa.eval_runs WHERE run_uid = :run_uid
                    )
                    ORDER BY i.id ASC
                    """
                ),
                {"run_uid": run_id},
            ).mappings().all()
        return normalize_eval_run(run, items=[normalize_eval_run_item(item) for item in items])

    def annotate_eval_item(self, *, item_id: str, label: str, comment: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.eval_run_items SET
                        annotation_label = :label,
                        annotation_comment = :comment,
                        annotated_at = CURRENT_TIMESTAMP
                    WHERE item_uid = :item_uid
                    RETURNING item_uid AS item_id, query, expected_intent, actual_intent,
                              expected_refused, actual_refused, expected_doc_ids,
                              retrieved_doc_ids, expected_answer_keywords, answer, audit_uid,
                              recall_hit, refusal_match, intent_match, keyword_hit,
                              annotation_label, annotation_comment,
                              '' AS case_id
                    """
                ),
                {"item_uid": item_id, "label": label, "comment": comment},
            ).mappings().first()
        if not row:
            raise ValueError("Evaluation run item not found")
        return normalize_eval_run_item(row)

    def persist_document(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> None:
        source_type = canonicalize_source_type(document.source_type)
        with self.engine.begin() as connection:
            kb_id = self._ensure_default_kb(connection)
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.documents (
                        document_uid, knowledge_base_id, title, filename, content_type,
                        source_type, source_url, raw_text, text_preview, content_hash,
                        biz_type, region_code, policy_version, material_code,
                        reject_code, audit_node, status, version_group_uid, version_no,
                        rollout_percent, change_note, effective_start_time, effective_end_time,
                        chunk_count
                    )
                    VALUES (
                        :document_uid, :knowledge_base_id, :title, :filename, :content_type,
                        :source_type, :source_url, :raw_text, :text_preview, :content_hash,
                        :biz_type, :region_code, :policy_version, :material_code,
                        :reject_code, :audit_node, :status, :version_group_uid, :version_no,
                        :rollout_percent, :change_note, :effective_start_time, :effective_end_time,
                        :chunk_count
                    )
                    ON CONFLICT (document_uid) WHERE document_uid <> '' DO UPDATE SET
                        title = EXCLUDED.title,
                        filename = EXCLUDED.filename,
                        content_type = EXCLUDED.content_type,
                        source_type = EXCLUDED.source_type,
                        source_url = EXCLUDED.source_url,
                        raw_text = EXCLUDED.raw_text,
                        text_preview = EXCLUDED.text_preview,
                        content_hash = EXCLUDED.content_hash,
                        biz_type = EXCLUDED.biz_type,
                        region_code = EXCLUDED.region_code,
                        policy_version = EXCLUDED.policy_version,
                        material_code = EXCLUDED.material_code,
                        reject_code = EXCLUDED.reject_code,
                        audit_node = EXCLUDED.audit_node,
                        status = EXCLUDED.status,
                        version_group_uid = EXCLUDED.version_group_uid,
                        version_no = EXCLUDED.version_no,
                        rollout_percent = EXCLUDED.rollout_percent,
                        change_note = EXCLUDED.change_note,
                        effective_start_time = EXCLUDED.effective_start_time,
                        effective_end_time = EXCLUDED.effective_end_time,
                        chunk_count = EXCLUDED.chunk_count,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                    """
                ),
                {
                    "document_uid": document.id,
                    "knowledge_base_id": kb_id,
                    "title": document.title,
                    "filename": document.filename,
                    "content_type": document.content_type,
                    "source_type": source_type,
                    "source_url": document.source_url,
                    "raw_text": document.content,
                    "text_preview": document.content[:240],
                    "content_hash": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
                    "biz_type": document.biz_type,
                    "region_code": document.region_code,
                    "policy_version": document.policy_version,
                    "material_code": document.material_code,
                    "reject_code": document.reject_code,
                    "audit_node": document.audit_node,
                    "status": document.status,
                    "version_group_uid": document.version_group_uid or document.id,
                    "version_no": document.version_no,
                    "rollout_percent": document.rollout_percent,
                    "change_note": document.change_note,
                    "effective_start_time": document.effective_start_time,
                    "effective_end_time": document.effective_end_time,
                    "chunk_count": len(chunks),
                },
            ).mappings().first()
            document_pk = int(row["id"])
            connection.execute(
                text("DELETE FROM declaration_qa.chunks WHERE document_id = :document_id"),
                {"document_id": document_pk},
            )
            parent_db_ids: dict[str, int] = {}
            for chunk in chunks:
                parent_pk = parent_db_ids.get(chunk.parent_id)
                chunk_row = self._insert_chunk(connection, document_pk, parent_pk, chunk)
                if chunk.chunk_type == "parent":
                    parent_db_ids[chunk.id] = int(chunk_row["id"])

    def list_documents(self) -> list[KnowledgeDocument]:
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT id, document_uid, title, filename, content_type, raw_text, biz_type,
                           region_code, policy_version, source_type, source_url, material_code,
                           reject_code, audit_node, status, version_group_uid, version_no,
                           rollout_percent, change_note, effective_start_time, effective_end_time
                    FROM declaration_qa.documents
                    WHERE status = 'published'
                    ORDER BY id ASC
                    """
                )
            ).mappings().all()
        return [
            KnowledgeDocument(
                id=str(row["document_uid"] or row["id"]),
                title=str(row["title"] or ""),
                filename=str(row["filename"] or ""),
                content_type=str(row["content_type"] or "text/plain"),
                content=str(row["raw_text"] or ""),
                biz_type=str(row["biz_type"] or ""),
                region_code=str(row["region_code"] or ""),
                policy_version=str(row["policy_version"] or ""),
                source_type=str(row["source_type"] or "guide"),
                source_url=str(row["source_url"] or ""),
                material_code=str(row["material_code"] or ""),
                reject_code=str(row["reject_code"] or ""),
                audit_node=str(row["audit_node"] or ""),
                status=str(row["status"] or "published"),
                version_group_uid=str(row["version_group_uid"] or row["document_uid"] or row["id"]),
                version_no=int(row["version_no"] or 1),
                rollout_percent=int(row["rollout_percent"] or 100),
                change_note=str(row["change_note"] or ""),
                effective_start_time=row["effective_start_time"],
                effective_end_time=row["effective_end_time"],
            )
            for row in rows
        ]

    def knowledge_base_revision(self) -> str:
        with self.engine.begin() as connection:
            value = connection.execute(
                text("SELECT COALESCE(MAX(updated_at)::text, '') FROM declaration_qa.documents")
            ).scalar_one()
        return str(value or "")

    def delete_document(self, document_uid: str) -> bool:
        with self.engine.begin() as connection:
            row = connection.execute(
                text("DELETE FROM declaration_qa.documents WHERE document_uid = :document_uid RETURNING id"),
                {"document_uid": document_uid},
            ).mappings().first()
        return bool(row)

    def list_document_records(self, *, status: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE status = :status" if status else ""
        params = {"status": status} if status else {}
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    f"""
                    SELECT document_uid AS document_id, title, filename, source_type, biz_type,
                           region_code, policy_version, material_code, reject_code, audit_node,
                           status, version_group_uid, version_no, rollout_percent,
                           effective_start_time, effective_end_time, chunk_count, change_note,
                           created_at, updated_at
                    FROM declaration_qa.documents
                    {where}
                    ORDER BY updated_at DESC, id DESC
                    """
                ),
                params,
            ).mappings().all()
        return [normalize_document_record(row) for row in rows]

    def get_document_record(self, document_uid: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT document_uid AS document_id, title, filename, source_type, biz_type,
                           region_code, policy_version, material_code, reject_code, audit_node,
                           status, version_group_uid, version_no, rollout_percent,
                           effective_start_time, effective_end_time, chunk_count, change_note,
                           raw_text, content_type, source_url, created_at, updated_at
                    FROM declaration_qa.documents
                    WHERE document_uid = :document_uid
                    """
                ),
                {"document_uid": document_uid},
            ).mappings().first()
        return dict(row) if row else None

    def set_document_status(self, document_uid: str, status: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.documents SET
                        status = :status,
                        published_at = CASE WHEN :status = 'published' THEN CURRENT_TIMESTAMP ELSE published_at END,
                        deprecated_at = CASE WHEN :status = 'deprecated' THEN CURRENT_TIMESTAMP ELSE deprecated_at END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_uid = :document_uid
                    RETURNING document_uid AS document_id, status
                    """
                ),
                {"document_uid": document_uid, "status": status},
            ).mappings().first()
            connection.execute(
                text(
                    """
                    UPDATE declaration_qa.chunks SET status = CASE
                        WHEN :status = 'published' THEN 'published'
                        WHEN :status = 'deprecated' THEN 'deprecated'
                        ELSE 'draft'
                    END
                    WHERE document_id = (SELECT id FROM declaration_qa.documents WHERE document_uid = :document_uid)
                    """
                ),
                {"document_uid": document_uid, "status": status},
            )
        if not row:
            raise ValueError("Document not found")
        return dict(row)

    def update_document_rollout(
        self,
        *,
        document_uid: str,
        rollout_percent: int,
        effective_start_time,
        effective_end_time,
    ) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.documents SET
                        rollout_percent = :rollout_percent,
                        effective_start_time = :effective_start_time,
                        effective_end_time = :effective_end_time,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_uid = :document_uid
                    RETURNING document_uid AS document_id, rollout_percent, effective_start_time, effective_end_time
                    """
                ),
                {
                    "document_uid": document_uid,
                    "rollout_percent": rollout_percent,
                    "effective_start_time": effective_start_time,
                    "effective_end_time": effective_end_time,
                },
            ).mappings().first()
        if not row:
            raise ValueError("Document not found")
        return dict(row)

    def create_document_version(
        self,
        *,
        source_document_uid: str,
        document_uid: str,
        title: str | None,
        content: str | None,
        policy_version: str | None,
        change_note: str,
    ) -> dict[str, Any]:
        source = self.get_document_record(source_document_uid)
        if not source:
            raise ValueError("Source document not found")
        group_uid = str(source.get("version_group_uid") or source_document_uid)
        with self.engine.begin() as connection:
            version_no = int(
                connection.execute(
                    text("SELECT COALESCE(MAX(version_no), 0) + 1 FROM declaration_qa.documents WHERE version_group_uid = :group_uid"),
                    {"group_uid": group_uid},
                ).scalar_one()
            )
        document = KnowledgeDocument(
            id=document_uid,
            title=title or str(source["title"]),
            filename=str(source.get("filename") or ""),
            content_type=str(source.get("content_type") or "text/plain"),
            content=content if content is not None else str(source.get("raw_text") or ""),
            biz_type=str(source["biz_type"]),
            region_code=str(source["region_code"]),
            policy_version=policy_version or str(source["policy_version"]),
            source_type=str(source["source_type"]),
            source_url=str(source.get("source_url") or ""),
            material_code=str(source.get("material_code") or ""),
            reject_code=str(source.get("reject_code") or ""),
            audit_node=str(source.get("audit_node") or ""),
            status="draft",
            version_group_uid=group_uid,
            version_no=version_no,
            rollout_percent=0,
            change_note=change_note,
            effective_start_time=source.get("effective_start_time"),
            effective_end_time=source.get("effective_end_time"),
        )
        self.persist_document(document, [])
        return {
            "document_id": document.id,
            "version_group_uid": group_uid,
            "version_no": version_no,
            "status": "draft",
        }

    def rollback_document(self, *, target_document_uid: str, change_note: str) -> dict[str, Any]:
        target = self.get_document_record(target_document_uid)
        if not target:
            raise ValueError("Target document not found")
        group_uid = str(target.get("version_group_uid") or target_document_uid)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE declaration_qa.documents
                    SET status = 'deprecated', deprecated_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE version_group_uid = :group_uid AND status = 'published'
                    """
                ),
                {"group_uid": group_uid},
            )
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.documents
                    SET status = 'published', change_note = :change_note,
                        published_at = CURRENT_TIMESTAMP, deprecated_at = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE document_uid = :target_document_uid
                    RETURNING document_uid AS document_id, status, version_group_uid, version_no
                    """
                ),
                {"target_document_uid": target_document_uid, "change_note": change_note},
            ).mappings().first()
        return dict(row)

    def create_index_job(self, *, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        from uuid import uuid4

        job_uid = uuid4().hex
        stmt = text(
            """
            INSERT INTO declaration_qa.index_jobs (job_uid, job_type, status, progress, payload)
            VALUES (:job_uid, :job_type, 'pending', 0, :payload)
            RETURNING *
            """
        ).bindparams(bindparam("payload", type_=JSON))
        with self.engine.begin() as connection:
            row = connection.execute(stmt, {"job_uid": job_uid, "job_type": job_type, "payload": payload}).mappings().first()
        return dict(row)

    def update_index_job(
        self,
        *,
        job_uid: str,
        status: str,
        progress: int,
        message: str = "",
        error_message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload_assignment = "payload = :payload," if payload is not None else ""
        stmt = text(
            f"""
            UPDATE declaration_qa.index_jobs SET
                status = CAST(:status AS VARCHAR),
                progress = :progress,
                message = :message,
                error_message = :error_message,
                {payload_assignment}
                started_at = CASE WHEN CAST(:status AS VARCHAR) = 'running' AND started_at IS NULL THEN CURRENT_TIMESTAMP ELSE started_at END,
                completed_at = CASE WHEN CAST(:status AS VARCHAR) IN ('succeeded', 'failed', 'cancelled') THEN CURRENT_TIMESTAMP ELSE completed_at END,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_uid = :job_uid
            RETURNING *
            """
        )
        if payload is not None:
            stmt = stmt.bindparams(bindparam("payload", type_=JSON))
        params = {
            "job_uid": job_uid,
            "status": status,
            "progress": max(0, min(100, int(progress))),
            "message": message,
            "error_message": error_message,
        }
        if payload is not None:
            params["payload"] = payload
        with self.engine.begin() as connection:
            row = connection.execute(stmt, params).mappings().first()
        if not row:
            raise ValueError("Index job not found")
        return dict(row)

    def get_index_job(self, job_uid: str) -> dict[str, Any] | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM declaration_qa.index_jobs WHERE job_uid = :job_uid"),
                {"job_uid": job_uid},
            ).mappings().first()
        return dict(row) if row else None

    def list_sessions(self, *, user_id: int) -> list[dict[str, Any]]:
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
        payload["messages"] = [dict(message) for message in messages]
        return payload

    def _ensure_session(self, connection, payload: dict[str, Any]) -> int:
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

    def _ensure_default_kb(self, connection) -> int:
        existing = connection.execute(
            text("SELECT id FROM declaration_qa.knowledge_bases WHERE name = '默认申报知识库' ORDER BY id ASC LIMIT 1")
        ).mappings().first()
        if existing:
            return int(existing["id"])
        row = connection.execute(
            text(
                """
                INSERT INTO declaration_qa.knowledge_bases (name, description)
                VALUES ('默认申报知识库', 'Declaration QA default knowledge base')
                RETURNING id
                """
            )
        ).mappings().first()
        return int(row["id"])

    def _insert_chunk(self, connection, document_id: int, parent_chunk_id: int | None, chunk: KnowledgeChunk):
        stmt = text(
            """
            INSERT INTO declaration_qa.chunks (
                document_id, parent_chunk_id, chunk_uid, chunk_type, content, content_hash,
                embedding_id, biz_type, region_code, policy_version, source_type, source_url,
                material_code, reject_code, audit_node, status, effective_start_time,
                effective_end_time, metadata
            )
            VALUES (
                :document_id, :parent_chunk_id, :chunk_uid, :chunk_type, :content, :content_hash,
                :embedding_id, :biz_type, :region_code, :policy_version, :source_type, :source_url,
                :material_code, :reject_code, :audit_node, :status, :effective_start_time,
                :effective_end_time, :metadata
            )
            ON CONFLICT (chunk_uid) DO UPDATE SET
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                embedding_id = EXCLUDED.embedding_id,
                biz_type = EXCLUDED.biz_type,
                region_code = EXCLUDED.region_code,
                policy_version = EXCLUDED.policy_version,
                source_type = EXCLUDED.source_type,
                source_url = EXCLUDED.source_url,
                material_code = EXCLUDED.material_code,
                reject_code = EXCLUDED.reject_code,
                audit_node = EXCLUDED.audit_node,
                status = EXCLUDED.status,
                effective_start_time = EXCLUDED.effective_start_time,
                effective_end_time = EXCLUDED.effective_end_time,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """
        ).bindparams(bindparam("metadata", type_=JSON))
        return connection.execute(
            stmt,
            {
                "document_id": document_id,
                "parent_chunk_id": parent_chunk_id,
                "chunk_uid": chunk.id,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "content_hash": hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
                "embedding_id": chunk.embedding_id,
                "biz_type": chunk.biz_type,
                "region_code": chunk.region_code,
                "policy_version": chunk.policy_version,
                "source_type": canonicalize_source_type(chunk.source_type),
                "source_url": chunk.source_url,
                "material_code": chunk.material_code,
                "reject_code": chunk.reject_code,
                "audit_node": chunk.audit_node,
                "status": chunk.status,
                "effective_start_time": chunk.effective_start_time,
                "effective_end_time": chunk.effective_end_time,
                "metadata": chunk.metadata,
            },
        ).mappings().first()


def normalize_session_row(row) -> dict[str, Any]:
    return {
        "session_id": str(row["session_id"] or ""),
        "user_id": int(row["user_id"] or 0),
        "region_code": str(row["region_code"] or ""),
        "biz_type": row["biz_type"],
        "apply_id": str(row["apply_id"] or ""),
        "title": str(row["title"] or "申报智能问答"),
        "message_count": int(row.get("message_count") or 0) if hasattr(row, "get") else int(row["message_count"] or 0),
        "updated_at": str(row["updated_at"] or ""),
    }


def normalize_document_record(row) -> dict[str, Any]:
    return {
        "document_id": str(row["document_id"] or ""),
        "title": str(row["title"] or ""),
        "filename": str(row["filename"] or ""),
        "source_type": str(row["source_type"] or ""),
        "biz_type": str(row["biz_type"] or ""),
        "region_code": str(row["region_code"] or ""),
        "policy_version": str(row["policy_version"] or ""),
        "material_code": str(row["material_code"] or ""),
        "reject_code": str(row["reject_code"] or ""),
        "audit_node": str(row["audit_node"] or ""),
        "status": str(row["status"] or ""),
        "version_group_uid": str(row["version_group_uid"] or row["document_id"] or ""),
        "version_no": int(row["version_no"] or 1),
        "rollout_percent": int(row["rollout_percent"] or 0),
        "effective_start_time": row["effective_start_time"].isoformat() if row["effective_start_time"] else None,
        "effective_end_time": row["effective_end_time"].isoformat() if row["effective_end_time"] else None,
        "chunk_count": int(row["chunk_count"] or 0),
        "change_note": str(row["change_note"] or ""),
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def normalize_business_row(row) -> dict[str, Any]:
    if not row:
        return {}
    values = dict(row)
    return {
        "apply_id": str(values.get("apply_id") or ""),
        "user_id": int(values.get("user_id") or 0) if values.get("user_id") is not None else None,
        "biz_type": str(values.get("biz_type") or ""),
        "region_code": str(values.get("region_code") or ""),
        "status": str(values.get("status") or ""),
        "current_node": str(values.get("current_node") or ""),
        "node_name": str(values.get("node_name") or ""),
        "reject_code": str(values.get("reject_code") or ""),
        "reject_reason": str(values.get("reject_reason") or ""),
        "material_code": str(values.get("material_code") or ""),
        "material_name": str(values.get("material_name") or ""),
        "audit_node": str(values.get("audit_node") or ""),
    }


def normalize_eval_dataset(row) -> dict[str, Any]:
    return {
        "dataset_id": str(row["dataset_id"] or ""),
        "name": str(row["name"] or ""),
        "description": str(row["description"] or ""),
        "case_count": int(row.get("case_count") or 0) if hasattr(row, "get") else 0,
        "created_at": str(row["created_at"] or ""),
    }


def normalize_eval_case(row) -> dict[str, Any]:
    return {
        "case_id": str(row["case_id"] or ""),
        "dataset_id": str(row["dataset_id"] or ""),
        "query": str(row["query"] or ""),
        "user_id": int(row["user_id"] or 1),
        "region_code": str(row["region_code"] or "340000"),
        "policy_version": str(row["policy_version"] or "2025"),
        "biz_type": row["biz_type"],
        "apply_id": row["apply_id"],
        "apply_context": row["apply_context"],
        "expected_intent": str(row["expected_intent"] or ""),
        "expected_refused": row["expected_refused"],
        "expected_doc_ids": list(row["expected_doc_ids"] or []),
        "expected_answer_keywords": list(row["expected_answer_keywords"] or []),
        "tags": list(row["tags"] or []),
        "created_at": str(row["created_at"] or ""),
    }


def normalize_eval_run(row, *, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": str(row["run_id"] or ""),
        "dataset_id": str(row["dataset_id"] or ""),
        "status": str(row["status"] or ""),
        "metrics": dict(row["metrics"] or {}),
        "items": items,
        "created_at": str(row["created_at"] or ""),
        "completed_at": str(row["completed_at"] or ""),
    }


def normalize_eval_run_item(row) -> dict[str, Any]:
    return {
        "item_id": str(row["item_id"] or ""),
        "case_id": str(row["case_id"] or ""),
        "query": str(row["query"] or ""),
        "expected_intent": str(row["expected_intent"] or ""),
        "actual_intent": str(row["actual_intent"] or ""),
        "expected_refused": row["expected_refused"],
        "actual_refused": bool(row["actual_refused"]),
        "expected_doc_ids": list(row["expected_doc_ids"] or []),
        "retrieved_doc_ids": list(row["retrieved_doc_ids"] or []),
        "expected_answer_keywords": list(row["expected_answer_keywords"] or []),
        "answer": str(row["answer"] or ""),
        "audit_id": str(row["audit_uid"] or ""),
        "recall_hit": row["recall_hit"],
        "refusal_match": row["refusal_match"],
        "intent_match": row["intent_match"],
        "keyword_hit": row["keyword_hit"],
        "annotation_label": str(row["annotation_label"] or ""),
        "annotation_comment": str(row["annotation_comment"] or ""),
    }
