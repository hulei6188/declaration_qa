"""LLM-based query rewriter."""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.query_rewriter.normalization import normalize_query_rewrite
from declaration_qa.rag.query_rewriter.prompts import build_query_rewriter_prompt
from declaration_qa.rag.query_rewriter.rule import RuleQueryRewriter
from declaration_qa.rag.query_rewriter.schema import QueryRewrite
from declaration_qa.rag.router import parse_json_object


class LLMQueryRewriter:
    """LLM-based multi-query rewrite with rule fallback."""

    name = "llm"

    def __init__(self, llm: Any, *, fallback_rewriter: RuleQueryRewriter | None = None) -> None:
        self.llm = llm
        self.fallback_rewriter = fallback_rewriter or RuleQueryRewriter()

    def rewrite(self, state: dict[str, Any]) -> QueryRewrite:
        """Call LLM to generate retrieval queries."""
        try:
            response = self.llm.invoke(build_query_rewriter_prompt(state))
            payload = parse_json_object(str(getattr(response, "content", response)))
            rewrite = normalize_query_rewrite(payload, state, provider=self.name)
            if not rewrite.queries:
                raise ValueError("empty_query_rewrite")
            return rewrite
        except Exception as exc:
            fallback = self.fallback_rewriter.rewrite(state)
            return QueryRewrite(
                base_query=fallback.base_query,
                scope_text=fallback.scope_text,
                fact_text=fallback.fact_text,
                requires_ticket_context=fallback.requires_ticket_context,
                ticket_fact_type=fallback.ticket_fact_type,
                queries=fallback.queries,
                provider="rule_fallback",
                reason=f"llm_query_rewriter_fallback:{type(exc).__name__}:{fallback.reason}",
            )
