"""Elasticsearch 关键词索引辅助函数。"""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.rag.retrieval import normalize_retrieval_filter


def elasticsearch_chunk_document(chunk: KnowledgeChunk) -> dict[str, Any]:
    """将 chunk 转换为可写入 Elasticsearch 的检索文档。"""
    metadata = dict(chunk.metadata or {})
    return {
        "doc_id": chunk.doc_id,
        "chunk_type": chunk.chunk_type,
        "title": chunk.title,
        "content": chunk.content,
        "region_code": chunk.region_code or metadata.get("region_code") or "",
        "region_key": metadata.get("region_key") or chunk.region_code or "",
        "policy_version": chunk.policy_version or metadata.get("policy_version") or "",
        "source_version": metadata.get("source_version") or chunk.policy_version or "",
        "catalog_name": metadata.get("catalog_name") or "",
        "catalog_subitem_name": metadata.get("catalog_subitem_name") or "",
        "item_code": metadata.get("item_code") or "",
        "item_name": metadata.get("item_name") or "",
        "item_type": metadata.get("item_type") or "",
        "ssqd_code": metadata.get("ssqd_code") or "",
        "ssqd_id": metadata.get("ssqd_id") or "",
        "theme_code": metadata.get("theme_code") or "",
        "service_object": metadata.get("service_object") or "",
        "audit_node": chunk.audit_node or metadata.get("audit_node") or "",
        "status": chunk.status or metadata.get("status") or "published",
        "doc_status": metadata.get("doc_status") or chunk.status or metadata.get("status") or "published",
        "metadata": metadata,
    }


def elasticsearch_filter_clauses(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """将标准化后的检索过滤条件转换为 Elasticsearch filter 子句。"""
    clauses: list[dict[str, Any]] = [{"term": {"chunk_type": "child"}}]
    for key, value in normalize_retrieval_filter(filters).items():
        if value in ("", None, []):
            continue
        if key in {"as_of", "rollout_key"}:
            continue
        field = elasticsearch_filter_field(key)
        if isinstance(value, list):
            values = [item for item in value if str(item or "").strip()]
            if values:
                clauses.append({"terms": {field: values}})
        else:
            clauses.append({"term": {field: value}})
    return clauses


def elasticsearch_filter_field(key: str) -> str:
    """返回过滤条件对应的精确匹配字段。"""
    if key in {"catalog_name", "catalog_subitem_name", "item_name"}:
        return f"{key}.keyword"
    if key == "doc_status":
        return "status"
    return key


def import_elasticsearch():
    """延迟导入 Elasticsearch SDK，确保该后端仍然是可选依赖。"""
    from elasticsearch import Elasticsearch, NotFoundError
    from elasticsearch.helpers import bulk

    return Elasticsearch, NotFoundError, bulk
