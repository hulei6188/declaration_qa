from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    content: str
    biz_type: str
    region_code: str
    policy_version: str
    source_type: str
    filename: str = ""
    content_type: str = "text/plain"
    source_url: str = ""
    material_code: str = ""
    reject_code: str = ""
    audit_node: str = ""
    status: str = "published"
    effective_start_time: date | None = None
    effective_end_time: date | None = None
    version_group_uid: str = ""
    version_no: int = 1
    rollout_percent: int = 100
    change_note: str = ""


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    doc_id: str
    parent_id: str
    chunk_type: str
    title: str
    content: str
    biz_type: str
    region_code: str
    policy_version: str
    source_type: str
    source_url: str = ""
    material_code: str = ""
    reject_code: str = ""
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
            "biz_type": self.chunk.biz_type,
            "region_code": self.chunk.region_code,
            "policy_version": self.chunk.policy_version,
            "source_type": self.chunk.source_type,
            "source_url": self.chunk.source_url,
            "material_code": self.chunk.material_code or None,
            "reject_code": self.chunk.reject_code or None,
            "audit_node": self.chunk.audit_node or None,
            "effective_start_time": self.chunk.effective_start_time.isoformat() if self.chunk.effective_start_time else None,
            "effective_end_time": self.chunk.effective_end_time.isoformat() if self.chunk.effective_end_time else None,
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
