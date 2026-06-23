"""Keyword retrieval indexes for declaration QA chunks."""

from __future__ import annotations

import json
from typing import Any, Protocol

from declaration_qa.domain import KnowledgeChunk
from declaration_qa.retrieval import normalize_retrieval_filter


class KeywordIndex(Protocol):
    """Keyword retrieval index protocol."""

    backend: str

    def upsert_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """Write searchable chunks to the keyword index."""
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        """Delete all indexed chunks for a document."""
        ...

    def search(self, query: str, filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """Search by keyword relevance and return chunk ids with scores."""
        ...

    def clear(self) -> None:
        """Clear all indexed chunks."""
        ...


class ElasticsearchKeywordIndex:
    """Elasticsearch-backed BM25 keyword index."""

    backend = "elasticsearch"

    def __init__(
        self,
        *,
        url: str,
        index_name: str,
        api_key: str = "",
        timeout_seconds: float = 10.0,
    ) -> None:
        """Initialize the Elasticsearch keyword index."""
        if not url:
            raise ValueError("Elasticsearch keyword index requires a URL")
        self.url = url.rstrip("/")
        self.index_name = index_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._ensure_index()

    def upsert_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """Write child chunks to Elasticsearch."""
        requests = import_requests()
        rows = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        if not rows:
            return
        payload_lines = []
        for chunk in rows:
            payload_lines.append(json.dumps({"index": {"_index": self.index_name, "_id": chunk.id}}, ensure_ascii=False))
            payload_lines.append(json.dumps(elasticsearch_chunk_document(chunk), ensure_ascii=False, default=str))
        response = requests.post(
            f"{self.url}/_bulk",
            data="\n".join(payload_lines) + "\n",
            headers={**self._headers(), "Content-Type": "application/x-ndjson"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def delete_by_doc(self, doc_id: str) -> None:
        """Delete all chunks for a document from Elasticsearch."""
        requests = import_requests()
        payload = {"query": {"term": {"doc_id": str(doc_id)}}}
        response = requests.post(
            f"{self.url}/{self.index_name}/_delete_by_query",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        if response.status_code == 404:
            return
        response.raise_for_status()

    def search(self, query: str, filters: dict[str, Any], *, limit: int) -> list[tuple[str, float]]:
        """Run a BM25 keyword search in Elasticsearch."""
        requests = import_requests()
        text = str(query or "").strip()
        if not text:
            return []
        payload = {
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
                    "filter": elasticsearch_filter_clauses(filters),
                }
            },
        }
        response = requests.post(
            f"{self.url}/{self.index_name}/_search",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        raw_scores = [float(item.get("_score") or 0.0) for item in hits if item.get("_id")]
        max_score = max(raw_scores or [0.0]) or 1.0
        return [
            (str(item.get("_id") or ""), float(item.get("_score") or 0.0) / max_score)
            for item in hits
            if item.get("_id")
        ]

    def clear(self) -> None:
        """Delete and recreate the Elasticsearch index."""
        requests = import_requests()
        response = requests.delete(
            f"{self.url}/{self.index_name}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        if response.status_code not in {200, 202, 404}:
            response.raise_for_status()
        self._ensure_index()

    def _ensure_index(self) -> None:
        requests = import_requests()
        response = requests.head(
            f"{self.url}/{self.index_name}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        if response.status_code == 200:
            return
        if response.status_code not in {404, 400}:
            response.raise_for_status()
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
                    "metadata": {"type": "object", "enabled": False},
                }
            }
        }
        create = requests.put(
            f"{self.url}/{self.index_name}",
            json=mapping,
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        create.raise_for_status()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        return headers


def elasticsearch_chunk_document(chunk: KnowledgeChunk) -> dict[str, Any]:
    """Convert a chunk into a searchable Elasticsearch document."""
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
        "metadata": metadata,
    }


def elasticsearch_filter_clauses(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert normalized retrieval filters into Elasticsearch filter clauses."""
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
    """Return the exact-match field used for a retrieval filter."""
    if key in {"catalog_name", "catalog_subitem_name", "item_name"}:
        return f"{key}.keyword"
    return key


def import_requests():
    """Import requests lazily so the Elasticsearch backend remains optional."""
    import requests

    return requests
