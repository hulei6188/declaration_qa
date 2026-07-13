"""PostgreSQL 存储行和领域对象转换。"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import text

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.rag.retrieval import sanitize_metadata

def normalize_session_row(row) -> dict[str, Any]:
    """标准化数据库会话行。"""
    return {
        "session_id": str(row["session_id"] or ""),
        "user_id": int(row["user_id"] or 0),
        "region_code": str(row["region_code"] or ""),
        "biz_type": row["biz_type"],
        "title": str(row["title"] or "申报智能问答"),
        "message_count": int(row.get("message_count") or 0) if hasattr(row, "get") else int(row["message_count"] or 0),
        "updated_at": str(row["updated_at"] or ""),
    }


def normalize_document_record(row) -> dict[str, Any]:
    """标准化知识文档管理记录。"""
    metadata = sanitize_metadata(dict(row.get("metadata") or {}) if hasattr(row, "get") else dict(row["metadata"] or {}))
    return {
        "document_id": str(row["document_id"] or ""),
        "title": str(row["title"] or ""),
        "filename": str(row["filename"] or ""),
        "region_code": str(metadata.get("region_code") or metadata.get("region_key") or ""),
        "policy_version": str(metadata.get("policy_version") or metadata.get("source_version") or ""),
        "audit_node": str(row["audit_node"] or ""),
        "status": str(row["status"] or ""),
        "version_group_uid": str(row["version_group_uid"] or row["document_id"] or ""),
        "version_no": int(row["version_no"] or 1),
        "rollout_percent": int(row["rollout_percent"] or 0),
        "effective_start_time": metadata.get("effective_start_time"),
        "effective_end_time": metadata.get("effective_end_time"),
        "chunk_count": int(row["chunk_count"] or 0),
        "change_note": str(row["change_note"] or ""),
        "metadata": metadata,
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def parse_metadata_date(value: Any) -> date | None:
    """从元数据中解析日期字段。"""
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def normalize_eval_dataset(row) -> dict[str, Any]:
    """标准化评测数据集行。"""
    return {
        "dataset_id": str(row["dataset_id"] or ""),
        "name": str(row["name"] or ""),
        "description": str(row["description"] or ""),
        "case_count": int(row.get("case_count") or 0) if hasattr(row, "get") else 0,
        "created_at": str(row["created_at"] or ""),
    }


def normalize_eval_case(row) -> dict[str, Any]:
    """标准化评测用例行。"""
    return {
        "case_id": str(row["case_id"] or ""),
        "dataset_id": str(row["dataset_id"] or ""),
        "query": str(row["query"] or ""),
        "user_id": int(row["user_id"] or 1),
        "region_code": str(row["region_code"] or "340000"),
        "policy_version": str(row["policy_version"] or "2025"),
        "biz_type": row["biz_type"],
        "expected_intent": str(row["expected_intent"] or ""),
        "expected_refused": row["expected_refused"],
        "expected_doc_ids": list(row["expected_doc_ids"] or []),
        "expected_answer_keywords": list(row["expected_answer_keywords"] or []),
        "reference_answer": str(row.get("reference_answer") or ""),
        "reference_contexts": list(row.get("reference_contexts") or []),
        "tags": list(row["tags"] or []),
        "created_at": str(row["created_at"] or ""),
    }


def normalize_eval_run(row, *, items: list[dict[str, Any]]) -> dict[str, Any]:
    """标准化评测运行行。"""
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
    """标准化评测运行明细行。"""
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
        "reference_answer": str(row.get("reference_answer") or ""),
        "reference_contexts": list(row.get("reference_contexts") or []),
        "retrieved_contexts": list(row.get("retrieved_contexts") or []),
        "answer": str(row["answer"] or ""),
        "audit_id": str(row["audit_uid"] or ""),
        "recall_hit": row["recall_hit"],
        "refusal_match": row["refusal_match"],
        "intent_match": row["intent_match"],
        "keyword_hit": row["keyword_hit"],
        "ragas_scores": dict(row.get("ragas_scores") or {}),
        "ragas_status": dict(row.get("ragas_status") or {}),
        "annotation_label": str(row["annotation_label"] or ""),
        "annotation_comment": str(row["annotation_comment"] or ""),
    }


def chunk_select_sql(where_clause: str):
    """构建包含文档和父级元数据的已发布片段查询语句。"""
    return text(
        f"""
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
          AND ({where_clause})
        ORDER BY d.id ASC, c.chunk_index ASC, c.id ASC
        """
    )


def knowledge_chunk_from_row(row) -> KnowledgeChunk:
    """将持久化片段行转换为 KnowledgeChunk。"""
    metadata = sanitize_metadata(dict(row["metadata"] or {}))
    return KnowledgeChunk(
        id=str(row["chunk_uid"] or ""),
        doc_id=str(row["document_uid"] or ""),
        parent_id=str(row["parent_chunk_uid"] or ""),
        chunk_type=str(row["chunk_type"] or ""),
        title=str(row["title"] or ""),
        content=str(row["content"] or ""),
        region_code=str(metadata.get("region_code") or metadata.get("region_key") or ""),
        policy_version=str(metadata.get("policy_version") or metadata.get("source_version") or ""),
        source_url=str(row["source_url"] or ""),
        audit_node=str(row["audit_node"] or ""),
        status=str(row["status"] or "published"),
        effective_start_time=parse_metadata_date(metadata.get("effective_start_time")),
        effective_end_time=parse_metadata_date(metadata.get("effective_end_time")),
        embedding_id=str(row["embedding_id"] or row["chunk_uid"] or ""),
        metadata=metadata,
    )


def ordered_chunks(rows, chunk_ids: list[str]) -> list[KnowledgeChunk]:
    """按请求的片段 ID 顺序返回片段。"""
    chunks = [knowledge_chunk_from_row(row) for row in rows]
    by_id = {chunk.id: chunk for chunk in chunks}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def unique_ordered_strings(values: list[str]) -> list[str]:
    """按首次出现顺序返回非空字符串。"""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
