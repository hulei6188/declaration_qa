from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    content: str
    region_code: str = ""
    policy_version: str = ""
    filename: str = ""
    content_type: str = "text/plain"
    source_url: str = ""
    audit_node: str = ""
    status: str = "published"
    effective_start_time: date | None = None
    effective_end_time: date | None = None
    version_group_uid: str = ""
    version_no: int = 1
    rollout_percent: int = 100
    change_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    doc_id: str
    parent_id: str
    chunk_type: str
    title: str
    content: str
    region_code: str = ""
    policy_version: str = ""
    source_url: str = ""
    audit_node: str = ""
    status: str = "published"
    effective_start_time: date | None = None
    effective_end_time: date | None = None
    embedding_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedSource:
    chunk: KnowledgeChunk
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_channel: str = "hybrid"

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "snippet": self.chunk.content[:500],
            "score": round(float(self.score), 4),
            "dense_score": round(float(self.dense_score), 4),
            "bm25_score": round(float(self.bm25_score), 4),
            "rerank_score": round(float(self.rerank_score), 4),
            "retrieval_channel": self.retrieval_channel,
            "region_code": self.chunk.region_code,
            "policy_version": self.chunk.policy_version,
            "region_key": self.chunk.metadata.get("region_key") or self.chunk.region_code,
            "region_name": self.chunk.metadata.get("region_name") or "",
            "catalog_name": self.chunk.metadata.get("catalog_name") or "",
            "catalog_subitem_name": self.chunk.metadata.get("catalog_subitem_name") or "",
            "item_code": self.chunk.metadata.get("item_code") or "",
            "item_name": self.chunk.metadata.get("item_name") or "",
            "item_type": self.chunk.metadata.get("item_type") or "",
            "theme_code": self.chunk.metadata.get("theme_code") or "",
            "service_object": self.chunk.metadata.get("service_object") or "",
            "source_version": self.chunk.metadata.get("source_version") or "",
            "source_url": self.chunk.source_url,
            "audit_node": self.chunk.audit_node or None,
            "effective_start_time": self.chunk.effective_start_time.isoformat() if self.chunk.effective_start_time else None,
            "effective_end_time": self.chunk.effective_end_time.isoformat() if self.chunk.effective_end_time else None,
            "metadata": dict(self.chunk.metadata or {}),
        }


@dataclass(frozen=True)
class RetrievalResult:
    sources: list[RetrievedSource]
    filter: dict[str, Any]
    status: dict[str, Any]


@dataclass(frozen=True)
class EvidenceDecision:
    ok: bool
    reason: str = ""
    top_score: float = 0.0
    source_count: int = 0
    required_metadata: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "top_score": round(float(self.top_score), 4),
            "source_count": self.source_count,
            "required_metadata": list(self.required_metadata),
            "missing_metadata": list(self.missing_metadata),
            "checks": dict(self.checks or {}),
        }
