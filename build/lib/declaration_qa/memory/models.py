"""上下文记忆领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from declaration_qa.core.domain import KnowledgeChunk


@dataclass(frozen=True)
class ContextMemoryItem:
    """可持久化和检索的长期上下文记忆。"""

    id: str
    user_id: int
    session_id: str
    memory_type: str
    content: str
    summary: str = ""
    source_audit_id: str = ""
    source_query: str = ""
    source_answer: str = ""
    region_code: str = ""
    biz_type: str = ""
    importance: float = 0.5
    status: str = "published"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_chunk(self) -> KnowledgeChunk:
        """转换为检索索引复用的知识片段结构。"""
        metadata = {
            **(self.metadata or {}),
            "memory_scope": "long_term",
            "memory_type": self.memory_type,
            "user_id": str(self.user_id),
            "session_id": self.session_id,
            "source_audit_id": self.source_audit_id,
            "source_query": self.source_query,
            "region_code": self.region_code,
            "biz_type": self.biz_type,
            "importance": self.importance,
            "doc_status": self.status,
            "modality": "text",
        }
        return KnowledgeChunk(
            id=self.id,
            doc_id=memory_doc_id(self.user_id),
            parent_id="",
            chunk_type="child",
            title=f"用户 {self.user_id} 长期上下文记忆",
            content=self.content,
            region_code=self.region_code,
            policy_version="memory",
            status=self.status,
            embedding_id=self.id,
            metadata=metadata,
        )

    def payload(self) -> dict[str, Any]:
        """转换为缓存和接口使用的字典。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "summary": self.summary,
            "source_audit_id": self.source_audit_id,
            "source_query": self.source_query,
            "source_answer": self.source_answer,
            "region_code": self.region_code,
            "biz_type": self.biz_type,
            "importance": self.importance,
            "status": self.status,
            "metadata": dict(self.metadata or {}),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ContextMemoryHit:
    """一次长期记忆检索命中。"""

    memory: ContextMemoryItem
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    retrieval_channel: str = "hybrid"

    def payload(self) -> dict[str, Any]:
        """转换为提示词和审计使用的字典。"""
        return {
            **self.memory.payload(),
            "score": round(float(self.score), 4),
            "dense_score": round(float(self.dense_score), 4),
            "bm25_score": round(float(self.bm25_score), 4),
            "retrieval_channel": self.retrieval_channel,
        }


def context_memory_from_payload(payload: dict[str, Any]) -> ContextMemoryItem:
    """从字典恢复长期记忆对象。"""
    return ContextMemoryItem(
        id=str(payload.get("id") or payload.get("memory_id") or ""),
        user_id=int(payload.get("user_id") or 0),
        session_id=str(payload.get("session_id") or ""),
        memory_type=str(payload.get("memory_type") or "qa_fact"),
        content=str(payload.get("content") or ""),
        summary=str(payload.get("summary") or ""),
        source_audit_id=str(payload.get("source_audit_id") or ""),
        source_query=str(payload.get("source_query") or ""),
        source_answer=str(payload.get("source_answer") or ""),
        region_code=str(payload.get("region_code") or ""),
        biz_type=str(payload.get("biz_type") or ""),
        importance=float(payload.get("importance") or 0.5),
        status=str(payload.get("status") or "published"),
        metadata=dict(payload.get("metadata") or {}),
        created_at=str(payload.get("created_at") or ""),
    )


def memory_doc_id(user_id: int) -> str:
    """返回用户长期记忆在索引中的文档域。"""
    return f"context-memory:user:{int(user_id)}"


def utc_now_text() -> str:
    """返回 UTC ISO 时间文本。"""
    return datetime.now(timezone.utc).isoformat()
