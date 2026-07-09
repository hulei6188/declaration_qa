"""查询改写数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_RETRIEVAL_TOP_K = 3
MAX_RETRIEVAL_QUERY_COUNT = 3


@dataclass(frozen=True)
class RetrievalQuery:
    """单条检索查询。"""

    id: str
    text: str
    strategy: str
    top_k: int = DEFAULT_RETRIEVAL_TOP_K

    def payload(self) -> dict[str, Any]:
        """转换为 API 和审计载荷。"""
        return {
            "id": self.id,
            "text": self.text,
            "strategy": self.strategy,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class QueryRewrite:
    """多查询改写结果。"""

    base_query: str
    scope_text: str
    fact_text: str
    requires_ticket_context: bool
    ticket_fact_type: str
    queries: list[RetrievalQuery]
    provider: str = "rule"
    reason: str = ""

    def payload(self) -> dict[str, Any]:
        """转换为 API 和审计载荷。"""
        return {
            "base_query": self.base_query,
            "scope_text": self.scope_text,
            "fact_text": self.fact_text,
            "requires_ticket_context": self.requires_ticket_context,
            "ticket_fact_type": self.ticket_fact_type,
            "queries": [query.payload() for query in self.queries],
            "provider": self.provider,
            "reason": self.reason,
        }
