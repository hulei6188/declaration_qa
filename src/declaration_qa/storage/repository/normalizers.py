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
        "apply_id": row.get("apply_id"),
        "apply_context": dict(row.get("apply_context") or {}) if row.get("apply_context") is not None else None,
        "title": str(row["title"] or "政务智办"),
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
        "version": int(row.get("version") or 1),
        "status": str(row.get("status") or "draft"),
        "source": str(row.get("source") or "manual"),
        "parent_dataset_id": str(row.get("parent_dataset_id") or "") or None,
        "knowledge_revision": str(row.get("knowledge_revision") or ""),
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
        "case_type": str(row.get("case_type") or "rag"),
        "turns": list(row.get("turns") or []),
        "fixture_key": str(row.get("fixture_key") or ""),
        "expected_chunk_ids": list(row.get("expected_chunk_ids") or []),
        "expected_trajectory": dict(row.get("expected_trajectory") or {}),
        "expected_outcome": dict(row.get("expected_outcome") or {}),
        "severity": str(row.get("severity") or "medium"),
        "enabled": bool(row.get("enabled", True)),
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
        "baseline_run_id": str(row.get("baseline_run_id") or "") or None,
        "trigger_source": str(row.get("trigger_source") or "manual"),
        "runtime_snapshot": dict(row.get("runtime_snapshot") or {}),
        "gate_result": dict(row.get("gate_result") or {}),
        "git_commit": str(row.get("git_commit") or ""),
        "error_message": str(row.get("error_message") or ""),
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
        "agent_run_uid": str(row.get("agent_run_uid") or ""),
        "actual_trajectory": dict(row.get("actual_trajectory") or {}),
        "rule_scores": dict(row.get("rule_scores") or {}),
        "judge_scores": dict(row.get("judge_scores") or {}),
        "failure_reasons": list(row.get("failure_reasons") or []),
        "passed": row.get("passed"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "total_tokens": row.get("total_tokens"),
        "latency_ms": row.get("latency_ms"),
        "estimated_cost": float(row["estimated_cost"]) if row.get("estimated_cost") is not None else None,
    }


def normalize_eval_candidate(row) -> dict[str, Any]:
    """标准化线上失败候选样本。"""
    return {
        "candidate_id": str(row.get("candidate_id") or ""),
        "audit_id": str(row.get("audit_uid") or ""),
        "agent_run_uid": str(row.get("agent_run_uid") or ""),
        "trace_id": str(row.get("trace_id") or ""),
        "source": str(row.get("source") or ""),
        "status": str(row.get("status") or "new"),
        "severity": str(row.get("severity") or "medium"),
        "sanitized_query": str(row.get("sanitized_query") or ""),
        "sanitized_answer": str(row.get("sanitized_answer") or ""),
        "detection_signals": dict(row.get("detection_signals") or {}),
        "trajectory_snapshot": dict(row.get("trajectory_snapshot") or {}),
        "labels": list(row.get("labels") or []),
        "reviewer_comment": str(row.get("reviewer_comment") or ""),
        "promoted_case_id": str(row.get("promoted_case_id") or "") or None,
        "created_at": str(row.get("created_at") or ""),
        "updated_at": str(row.get("updated_at") or ""),
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
