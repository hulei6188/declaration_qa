"""Rule-based query rewriter."""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.query_rewriter.helpers import (
    INTENT_REWRITE_TERMS,
    has_item_scope_filter,
    join_query_parts,
    normalize_text,
    selected_ticket_fact_text,
    selected_ticket_scope_text,
)
from declaration_qa.rag.query_rewriter.schema import DEFAULT_RETRIEVAL_TOP_K, MAX_RETRIEVAL_QUERY_COUNT, QueryRewrite, RetrievalQuery


class RuleQueryRewriter:
    """Rule-based multi-query rewrite for stable retrieval recall."""

    name = "rule"

    def rewrite(self, state: dict[str, Any]) -> QueryRewrite:
        """Generate retrieval queries from task semantics and item scope."""
        base_query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
        retrieval_focus = str(state.get("retrieval_focus") or state.get("intent") or "other")
        ticket_fact_type = str(state.get("ticket_fact_type") or "none")
        requires_ticket_context = bool(state.get("requires_ticket_context"))
        scope_text = "" if has_item_scope_filter(state.get("retrieval_filter") or {}) else selected_ticket_scope_text(state)
        fact_text = selected_ticket_fact_text(state) if requires_ticket_context else ""
        candidates = [("scoped_original", join_query_parts(scope_text, fact_text, base_query))]
        for index, terms in enumerate(INTENT_REWRITE_TERMS.get(retrieval_focus, INTENT_REWRITE_TERMS["other"]), start=1):
            candidates.append((f"intent_terms_{index}", join_query_parts(scope_text, fact_text, terms, base_query)))
        queries: list[RetrievalQuery] = []
        seen = set()
        for strategy, text in candidates:
            normalized = normalize_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            queries.append(
                RetrievalQuery(
                    id=f"rq{len(queries) + 1}",
                    text=normalized,
                    strategy=strategy,
                    top_k=DEFAULT_RETRIEVAL_TOP_K,
                )
            )
            if len(queries) >= MAX_RETRIEVAL_QUERY_COUNT:
                break
        return QueryRewrite(
            base_query=base_query,
            scope_text=scope_text,
            fact_text=fact_text,
            requires_ticket_context=requires_ticket_context,
            ticket_fact_type=ticket_fact_type,
            retrieval_focus=retrieval_focus,
            queries=queries,
            provider=self.name,
            reason="rule_rewrite",
        )
