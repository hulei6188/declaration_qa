"""申报问答知识 chunk 的关键词检索索引。"""

from __future__ import annotations

from typing import Any, Protocol

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.rag.retrieval import normalize_retrieval_filter


class KeywordIndex(Protocol):
    """关键词检索索引协议。"""

    backend: str

    def upsert_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """将可检索的 chunk 写入关键词索引。"""
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        """删除指定文档下已经索引的全部 chunk。"""
        ...

    def search(self, query: str, filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """按关键词相关性检索，返回 chunk id 及归一化分数。"""
        ...

    def search_many(
        self,
        query_specs: list[dict[str, Any]],
        filters: dict[str, Any],
        limits: list[int],
    ) -> dict[int, list[tuple[str, float]]]:
        """批量关键词检索，返回每个 query index 对应的 chunk id 和分数。"""
        ...

    def clear(self) -> None:
        """清空全部关键词索引数据。"""
        ...


class ElasticsearchKeywordIndex:
    """基于 Elasticsearch 的 BM25 关键词索引。"""

    backend = "elasticsearch"

    def __init__(
        self,
        *,
        url: str,
        index_name: str,
        api_key: str = "",
        timeout_seconds: float = 10.0,
    ) -> None:
        """初始化 Elasticsearch 关键词索引。"""
        if not url:
            raise ValueError("Elasticsearch 关键词索引需要配置 URL")
        self.url = url.rstrip("/")
        self.index_name = index_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        Elasticsearch, NotFoundError, bulk = import_elasticsearch()
        client_kwargs: dict[str, Any] = {"request_timeout": self.timeout_seconds}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        self._client = Elasticsearch(self.url, **client_kwargs)
        self._not_found_error = NotFoundError
        self._bulk = bulk
        self._ensure_index()

    def upsert_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """将 child chunk 写入 Elasticsearch。"""
        rows = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        if not rows:
            return
        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": chunk.id,
                "_source": elasticsearch_chunk_document(chunk),
            }
            for chunk in rows
        ]
        self._bulk(self._client, actions, refresh=True)

    def delete_by_doc(self, doc_id: str) -> None:
        """从 Elasticsearch 删除指定文档下的全部 chunk。"""
        try:
            self._client.delete_by_query(
                index=self.index_name,
                query={"term": {"doc_id": str(doc_id)}},
                refresh=True,
            )
        except self._not_found_error:
            return

    def search(self, query: str, filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """在 Elasticsearch 中执行 BM25 关键词检索。"""
        results = self.search_many([{"index": 0, "query": query}], filters, [limit])
        return results.get(0, [])

    def search_many(
        self,
        query_specs: list[dict[str, Any]],
        filters: dict[str, Any],
        limits: list[int],
    ) -> dict[int, list[tuple[str, float]]]:
        """使用 Elasticsearch msearch 批量执行 BM25 关键词检索。"""
        results: dict[int, list[tuple[str, float]]] = {
            int(spec.get("index") or 0): []
            for spec in query_specs
        }
        if not query_specs:
            return results
        filter_clauses = elasticsearch_filter_clauses(filters)
        searches: list[dict[str, Any]] = []
        request_indexes: list[int] = []
        for spec, limit in zip(query_specs, limits):
            index = int(spec.get("index") or 0)
            text = str(spec.get("query") or spec.get("text") or "").strip()
            if not text:
                continue
            searches.append({"index": self.index_name})
            searches.append(
                {
                    "size": max(1, int(limit)),
                    "_source": False,
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "multi_match": {
                                        "query": text,
                                        "fields": ["content^3", "title^2", "item_name^2", "catalog_name"],
                                        "type": "best_fields",
                                    }
                                }
                            ],
                            "filter": filter_clauses,
                        }
                    },
                }
            )
            request_indexes.append(index)
        if not searches:
            return results
        try:
            response = self._client.msearch(searches=searches)
        except TypeError:
            response = self._client.msearch(body=searches)
        responses = response.get("responses", [])
        for index, item in zip(request_indexes, responses):
            if item.get("error"):
                results[index] = []
                continue
            hits = item.get("hits", {}).get("hits", [])
            raw_scores = [float(hit.get("_score") or 0.0) for hit in hits if hit.get("_id")]
            max_score = max(raw_scores or [0.0]) or 1.0
            results[index] = [
                (str(hit.get("_id") or ""), float(hit.get("_score") or 0.0) / max_score)
                for hit in hits
                if hit.get("_id")
            ]
        return results

    def clear(self) -> None:
        """删除并重新创建 Elasticsearch 索引。"""
        try:
            self._client.indices.delete(index=self.index_name)
        except self._not_found_error:
            pass
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self._client.indices.exists(index=self.index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "region_code": {"type": "keyword"},
                    "region_key": {"type": "keyword"},
                    "policy_version": {"type": "keyword"},
                    "source_version": {"type": "keyword"},
                    "catalog_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "catalog_subitem_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "item_code": {"type": "keyword"},
                    "item_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "item_type": {"type": "keyword"},
                    "ssqd_code": {"type": "keyword"},
                    "ssqd_id": {"type": "keyword"},
                    "theme_code": {"type": "keyword"},
                    "service_object": {"type": "keyword"},
                    "audit_node": {"type": "keyword"},
                    "status": {"type": "keyword"},
                    "doc_status": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                }
            }
        }
        self._client.indices.create(
            index=self.index_name,
            mappings=mapping["mappings"],
        )


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
