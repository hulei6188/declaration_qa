"""知识文档、chunk store 和版本管理存储。"""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import JSON, bindparam, text

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument
from declaration_qa.rag.retrieval import sanitize_metadata
from declaration_qa.storage.repository.base import EngineBackedRepository
from declaration_qa.storage.repository.normalizers import (
    chunk_select_sql,
    knowledge_chunk_from_row,
    normalize_document_record,
    ordered_chunks,
    parse_metadata_date,
    unique_ordered_strings,
)


class KnowledgeRepository(EngineBackedRepository):
    """知识文档持久化、chunk 读取和版本管理仓储。"""
    def persist_document(self, document: KnowledgeDocument, chunks: list[KnowledgeChunk]) -> None:
        """持久化知识文档及其片段。"""
        with self.engine.begin() as connection:
            kb_id = self._ensure_default_kb(connection)
            row = connection.execute(
                text(
                    """
                    INSERT INTO declaration_qa.documents (
                        document_uid, knowledge_base_id, title, filename, content_type,
                        source_url, raw_text, text_preview, content_hash,
                        audit_node, status, version_group_uid, version_no,
                        rollout_percent, change_note,
                        chunk_count, metadata
                    )
                    VALUES (
                        :document_uid, :knowledge_base_id, :title, :filename, :content_type,
                        :source_url, :raw_text, :text_preview, :content_hash,
                        :audit_node, :status, :version_group_uid, :version_no,
                        :rollout_percent, :change_note,
                        :chunk_count, :metadata
                    )
                    ON CONFLICT (document_uid) WHERE document_uid <> '' DO UPDATE SET
                        title = EXCLUDED.title,
                        filename = EXCLUDED.filename,
                        content_type = EXCLUDED.content_type,
                        source_url = EXCLUDED.source_url,
                        raw_text = EXCLUDED.raw_text,
                        text_preview = EXCLUDED.text_preview,
                        content_hash = EXCLUDED.content_hash,
                        audit_node = EXCLUDED.audit_node,
                        status = EXCLUDED.status,
                        version_group_uid = EXCLUDED.version_group_uid,
                        version_no = EXCLUDED.version_no,
                        rollout_percent = EXCLUDED.rollout_percent,
                        change_note = EXCLUDED.change_note,
                        chunk_count = EXCLUDED.chunk_count,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                    """
                ).bindparams(bindparam("metadata", type_=JSON)),
                {
                    "document_uid": document.id,
                    "knowledge_base_id": kb_id,
                    "title": document.title,
                    "filename": document.filename,
                    "content_type": document.content_type,
                    "source_url": document.source_url,
                    "raw_text": document.content,
                    "text_preview": document.content[:240],
                    "content_hash": hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
                    "audit_node": document.audit_node,
                    "status": document.status,
                    "version_group_uid": document.version_group_uid or document.id,
                    "version_no": document.version_no,
                    "rollout_percent": document.rollout_percent,
                    "change_note": document.change_note,
                    "chunk_count": len(chunks),
                    "metadata": sanitize_metadata(document.metadata),
                },
            ).mappings().first()
            document_pk = int(row["id"])
            connection.execute(
                text("DELETE FROM declaration_qa.assets WHERE document_id = :document_id"),
                {"document_id": document_pk},
            )
            connection.execute(
                text("DELETE FROM declaration_qa.chunks WHERE document_id = :document_id"),
                {"document_id": document_pk},
            )
            parent_db_ids: dict[str, int] = {}
            for chunk in chunks:
                parent_pk = parent_db_ids.get(chunk.parent_id)
                chunk_row = self._insert_chunk(connection, document_pk, parent_pk, chunk)
                chunk_pk = int(chunk_row["id"])
                if chunk.chunk_type == "parent":
                    parent_db_ids[chunk.id] = chunk_pk
                self._insert_asset_if_present(connection, kb_id, document_pk, chunk_pk, chunk)

    def list_documents(self) -> list[KnowledgeDocument]:
        """查询已发布或可索引的知识文档。"""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT id, document_uid, title, filename, content_type, raw_text,
                           source_url, audit_node, status, version_group_uid, version_no,
                           rollout_percent, change_note, metadata
                    FROM declaration_qa.documents
                    WHERE status = 'published'
                    ORDER BY id ASC
                    """
                )
            ).mappings().all()
        documents = []
        for row in rows:
            metadata = sanitize_metadata(dict(row["metadata"] or {}))
            documents.append(KnowledgeDocument(
                id=str(row["document_uid"] or row["id"]),
                title=str(row["title"] or ""),
                filename=str(row["filename"] or ""),
                content_type=str(row["content_type"] or "text/plain"),
                content=str(row["raw_text"] or ""),
                region_code=str(metadata.get("region_code") or metadata.get("region_key") or ""),
                policy_version=str(metadata.get("policy_version") or metadata.get("source_version") or ""),
                source_url=str(row["source_url"] or ""),
                audit_node=str(row["audit_node"] or ""),
                status=str(row["status"] or "published"),
                version_group_uid=str(row["version_group_uid"] or row["document_uid"] or row["id"]),
                version_no=int(row["version_no"] or 1),
                rollout_percent=int(row["rollout_percent"] or 100),
                change_note=str(row["change_note"] or ""),
                effective_start_time=parse_metadata_date(metadata.get("effective_start_time")),
                effective_end_time=parse_metadata_date(metadata.get("effective_end_time")),
                metadata=metadata,
            ))
        return documents

    def list_indexed_knowledge(self) -> tuple[list[KnowledgeDocument], list[KnowledgeChunk]]:
        """查询已持久化的知识文档和片段。"""
        documents = self.list_documents()
        document_ids = {document.id for document in documents}
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        c.chunk_uid,
                        COALESCE(d.document_uid, d.id::text) AS document_uid,
                        COALESCE(p.chunk_uid, '') AS parent_chunk_uid,
                        c.chunk_type,
                        d.title,
                        c.content,
                        c.embedding_id,
                        COALESCE(c.source_url, d.source_url, '') AS source_url,
                        COALESCE(c.audit_node, d.audit_node, '') AS audit_node,
                        c.status,
                        c.metadata
                    FROM declaration_qa.chunks c
                    JOIN declaration_qa.documents d ON d.id = c.document_id
                    LEFT JOIN declaration_qa.chunks p ON p.id = c.parent_chunk_id
                    WHERE d.status = 'published'
                      AND c.status = 'published'
                    ORDER BY d.id ASC, c.chunk_index ASC, c.id ASC
                    """
                )
            ).mappings().all()
        chunks = []
        for row in rows:
            doc_id = str(row["document_uid"] or "")
            if document_ids and doc_id not in document_ids:
                continue
            chunks.append(knowledge_chunk_from_row(row))
        return documents, chunks

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        """按公开 chunk_uid 批量查询知识片段。"""
        ids = unique_ordered_strings(chunk_ids)
        if not ids:
            return []
        with self.engine.begin() as connection:
            rows = connection.execute(
                chunk_select_sql("c.chunk_uid = ANY(:chunk_ids)"),
                {"chunk_ids": ids},
            ).mappings().all()
        return ordered_chunks(rows, ids)

    def get_parent_chunks(self, parent_ids: list[str]) -> list[KnowledgeChunk]:
        """按公开 parent chunk_uid 批量查询父级片段。"""
        return self.get_chunks_by_ids(parent_ids)

    def get_structured_section_chunks(self, parent_section_ids: list[str], target_chunk_type: str) -> list[KnowledgeChunk]:
        """按结构化章节批量查询同级材料/流程片段。"""
        section_ids = unique_ordered_strings(parent_section_ids)
        if not section_ids or not str(target_chunk_type or "").strip():
            return []
        with self.engine.begin() as connection:
            rows = connection.execute(
                chunk_select_sql(
                    """
                    c.chunk_type = 'child'
                    AND COALESCE(c.metadata ->> 'indexable', 'true') <> 'false'
                    AND c.metadata ->> 'parent_section_id' = ANY(:parent_section_ids)
                    AND c.metadata ->> 'chunk_type' = :target_chunk_type
                    """
                ),
                {"parent_section_ids": section_ids, "target_chunk_type": str(target_chunk_type)},
            ).mappings().all()
        return [knowledge_chunk_from_row(row) for row in rows]

    def knowledge_base_revision(self) -> str:
        """返回知识库当前修订标识。"""
        with self.engine.begin() as connection:
            value = connection.execute(
                text("SELECT COALESCE(MAX(updated_at)::text, '') FROM declaration_qa.documents")
            ).scalar_one()
        return str(value or "")

    def delete_document(self, document_uid: str) -> bool:
        """删除知识文档及相关片段。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text("DELETE FROM declaration_qa.documents WHERE document_uid = :document_uid RETURNING id"),
                {"document_uid": document_uid},
            ).mappings().first()
        return bool(row)

    def list_document_records(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """查询后台管理用知识文档记录。"""
        where = "WHERE status = :status" if status else ""
        params = {"status": status} if status else {}
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    f"""
                    SELECT document_uid AS document_id, title, filename,
                           audit_node, status, version_group_uid, version_no, rollout_percent,
                           chunk_count, change_note, metadata,
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
        """查询单条后台管理用知识文档记录。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT document_uid AS document_id, title, filename,
                           audit_node, status, version_group_uid, version_no, rollout_percent,
                           chunk_count, change_note,
                           raw_text, content_type, source_url, metadata, created_at, updated_at
                    FROM declaration_qa.documents
                    WHERE document_uid = :document_uid
                    """
                ),
                {"document_uid": document_uid},
            ).mappings().first()
        return dict(row) if row else None

    def set_document_status(self, document_uid: str, status: str) -> dict[str, Any]:
        """更新知识文档发布状态。"""
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
        """调整知识文档灰度比例。"""
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    UPDATE declaration_qa.documents SET
                        rollout_percent = :rollout_percent,
                        metadata = jsonb_strip_nulls(
                            metadata
                            || jsonb_build_object(
                                'effective_start_time', to_jsonb(CAST(:effective_start_time AS TEXT)),
                                'effective_end_time', to_jsonb(CAST(:effective_end_time AS TEXT))
                            )
                        ),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE document_uid = :document_uid
                    RETURNING document_uid AS document_id, rollout_percent, metadata
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
        payload = dict(row)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        payload["effective_start_time"] = metadata.get("effective_start_time")
        payload["effective_end_time"] = metadata.get("effective_end_time")
        return payload

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
        """基于已有文档创建新版本。"""
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
        metadata = sanitize_metadata(dict(source.get("metadata") or {}))
        if policy_version:
            metadata["policy_version"] = policy_version
        document = KnowledgeDocument(
            id=document_uid,
            title=title or str(source["title"]),
            filename=str(source.get("filename") or ""),
            content_type=str(source.get("content_type") or "text/plain"),
            content=content if content is not None else str(source.get("raw_text") or ""),
            region_code=str(metadata.get("region_code") or metadata.get("region_key") or ""),
            policy_version=str(metadata.get("policy_version") or metadata.get("source_version") or ""),
            source_url=str(source.get("source_url") or ""),
            audit_node=str(source.get("audit_node") or ""),
            status="draft",
            version_group_uid=group_uid,
            version_no=version_no,
            rollout_percent=0,
            change_note=change_note,
            effective_start_time=parse_metadata_date(metadata.get("effective_start_time")),
            effective_end_time=parse_metadata_date(metadata.get("effective_end_time")),
            metadata=metadata,
        )
        self.persist_document(document, [])
        return {
            "document_id": document.id,
            "version_group_uid": group_uid,
            "version_no": version_no,
            "status": "draft",
        }

    def rollback_document(self, *, target_document_uid: str, change_note: str) -> dict[str, Any]:
        """将文档版本组回滚到指定版本。"""
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

    def _ensure_default_kb(self, connection) -> int:
        """确保默认知识库记录存在。"""
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
        """插入单个知识片段记录。"""
        stmt = text(
            """
            INSERT INTO declaration_qa.chunks (
                document_id, parent_chunk_id, chunk_uid, chunk_type, content, content_hash,
                embedding_id, source_url, audit_node, status, metadata
            )
            VALUES (
                :document_id, :parent_chunk_id, :chunk_uid, :chunk_type, :content, :content_hash,
                :embedding_id, :source_url, :audit_node, :status, :metadata
            )
            ON CONFLICT (chunk_uid) DO UPDATE SET
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                embedding_id = EXCLUDED.embedding_id,
                source_url = EXCLUDED.source_url,
                audit_node = EXCLUDED.audit_node,
                status = EXCLUDED.status,
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
                "source_url": chunk.source_url,
                "audit_node": chunk.audit_node,
                "status": chunk.status,
                "metadata": chunk.metadata,
            },
        ).mappings().first()

    def _insert_asset_if_present(
        self,
        connection,
        knowledge_base_id: int,
        document_id: int,
        chunk_id: int,
        chunk: KnowledgeChunk,
    ) -> None:
        """从图片资产 chunk 元数据中登记资产记录。"""
        metadata = dict(chunk.metadata or {})
        asset_uid = str(metadata.get("asset_id") or "").strip()
        if not asset_uid:
            return
        stmt = text(
            """
            INSERT INTO declaration_qa.assets (
                asset_uid, knowledge_base_id, document_id, chunk_id,
                attachment_index, attachment_name, attachment_type, attachment_role, attachment_sha256,
                asset_kind, modality, filename, content_type, file_path, asset_url, thumbnail_url,
                page_no, bbox, ocr_text, extraction_status, error_message, metadata
            )
            VALUES (
                :asset_uid, :knowledge_base_id, :document_id, :chunk_id,
                :attachment_index, :attachment_name, :attachment_type, :attachment_role, :attachment_sha256,
                :asset_kind, :modality, :filename, :content_type, :file_path, :asset_url, :thumbnail_url,
                :page_no, :bbox, :ocr_text, :extraction_status, :error_message, :metadata
            )
            ON CONFLICT (asset_uid) DO UPDATE SET
                knowledge_base_id = EXCLUDED.knowledge_base_id,
                document_id = EXCLUDED.document_id,
                chunk_id = EXCLUDED.chunk_id,
                attachment_index = EXCLUDED.attachment_index,
                attachment_name = EXCLUDED.attachment_name,
                attachment_type = EXCLUDED.attachment_type,
                attachment_role = EXCLUDED.attachment_role,
                attachment_sha256 = EXCLUDED.attachment_sha256,
                asset_kind = EXCLUDED.asset_kind,
                modality = EXCLUDED.modality,
                filename = EXCLUDED.filename,
                content_type = EXCLUDED.content_type,
                file_path = EXCLUDED.file_path,
                asset_url = EXCLUDED.asset_url,
                thumbnail_url = EXCLUDED.thumbnail_url,
                page_no = EXCLUDED.page_no,
                bbox = EXCLUDED.bbox,
                ocr_text = EXCLUDED.ocr_text,
                extraction_status = EXCLUDED.extraction_status,
                error_message = EXCLUDED.error_message,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
            """
        ).bindparams(bindparam("bbox", type_=JSON), bindparam("metadata", type_=JSON))
        connection.execute(
            stmt,
            {
                "asset_uid": asset_uid,
                "knowledge_base_id": knowledge_base_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "attachment_index": non_negative_int(metadata.get("attachment_index")),
                "attachment_name": text_value(metadata.get("attachment_name"), limit=255),
                "attachment_type": text_value(metadata.get("attachment_type"), limit=40),
                "attachment_role": text_value(metadata.get("attachment_role"), limit=80),
                "attachment_sha256": text_value(metadata.get("attachment_sha256"), limit=80),
                "asset_kind": text_value(metadata.get("asset_kind"), limit=80),
                "modality": text_value(metadata.get("modality"), limit=40),
                "filename": text_value(metadata.get("asset_filename"), limit=255),
                "content_type": text_value(metadata.get("asset_content_type"), limit=120),
                "file_path": text_value(metadata.get("asset_path")),
                "asset_url": text_value(metadata.get("asset_url")),
                "thumbnail_url": text_value(metadata.get("thumbnail_url")),
                "page_no": positive_int_or_none(metadata.get("attachment_page_no")),
                "bbox": metadata.get("bbox") if isinstance(metadata.get("bbox"), list) else None,
                "ocr_text": text_value(metadata.get("asset_text")),
                "extraction_status": asset_extraction_status(metadata.get("attachment_status")),
                "error_message": text_value(metadata.get("visual_asset_error") or metadata.get("extraction_error")),
                "metadata": sanitize_metadata({key: value for key, value in metadata.items() if key != "asset_text"}),
            },
        )


def text_value(value: Any, *, limit: int | None = None) -> str:
    """将元数据值归一化为可写入数据库的文本。"""
    text_value = str(value or "")
    if limit is not None:
        return text_value[:limit]
    return text_value


def non_negative_int(value: Any) -> int:
    """将元数据值归一化为非负整数。"""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def positive_int_or_none(value: Any) -> int | None:
    """将元数据值归一化为正整数。"""
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def asset_extraction_status(value: Any) -> str:
    """归一化资产抽取状态。"""
    status = str(value or "success").strip().lower()
    if status in {"success", "failed", "skipped"}:
        return status
    return "success"

