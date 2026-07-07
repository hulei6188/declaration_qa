"""Query rewrite data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_RETRIEVAL_TOP_K = 3
MAX_RETRIEVAL_QUERY_COUNT = 3


@dataclass(frozen=True)
class RetrievalQuery:
    """Single retrieval query."""

    id: str
    text: str
    strategy: str
    top_k: int = DEFAULT_RETRIEVAL_TOP_K

    def payload(self) -> dict[str, Any]:
        """Convert to API/audit payload."""
        return {
            "id": self.id,
            "text": self.text,
            "strategy": self.strategy,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class QueryRewrite:
    """Multi-query rewrite result."""

    base_query: str
    scope_text: str
    fact_text: str
    requires_ticket_context: bool
    ticket_fact_type: str
    retrieval_focus: str
    queries: list[RetrievalQuery]
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    provider: str = "rule"
    reason: str = ""

    def payload(self) -> dict[str, Any]:
        """Convert to API/audit payload."""
        return {
            "base_query": self.base_query,
            "scope_text": self.scope_text,
            "fact_text": self.fact_text,
            "requires_ticket_context": self.requires_ticket_context,
            "ticket_fact_type": self.ticket_fact_type,
            "retrieval_focus": self.retrieval_focus,
            "queries": [query.payload() for query in self.queries],
            "retrieval_plan": dict(self.retrieval_plan),
            "provider": self.provider,
            "reason": self.reason,
        }
