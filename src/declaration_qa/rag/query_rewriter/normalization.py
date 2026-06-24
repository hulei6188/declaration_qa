"""Normalize LLM query rewrite outputs."""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.query_rewriter.helpers import (
    has_item_scope_filter,
    normalize_strategy_name,
    normalize_text,
    normalize_top_k,
    selected_ticket_fact_text,
    selected_ticket_scope_text,
)
from declaration_qa.rag.query_rewriter.schema import DEFAULT_RETRIEVAL_TOP_K, MAX_RETRIEVAL_QUERY_COUNT, QueryRewrite, RetrievalQuery


def normalize_query_rewrite(payload: dict[str, Any], state: dict[str, Any], *, provider: str) -> QueryRewrite:
    """Normalize LLM query rewrite payload."""
    base_query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
    retrieval_focus = str(state.get("retrieval_focus") or state.get("intent") or "other")
    ticket_fact_type = str(state.get("ticket_fact_type") or "none")
    requires_ticket_context = bool(state.get("requires_ticket_context"))
    filters = state.get("retrieval_filter") or {}
    scope_text = "" if has_item_scope_filter(filters) else selected_ticket_scope_text(state)
    fact_text = selected_ticket_fact_text(state) if requires_ticket_context else ""
    queries: list[RetrievalQuery] = []
    seen = set()
    raw_queries = payload.get("queries") if isinstance(payload.get("queries"), list) else []
    for item in raw_queries:
        if isinstance(item, dict):
            text = normalize_text(str(item.get("text") or item.get("query") or ""))
            strategy = normalize_strategy_name(item.get("strategy") or "llm_rewrite")
            top_k = normalize_top_k(item.get("top_k"))
        else:
            text = normalize_text(str(item or ""))
            strategy = "llm_rewrite"
            top_k = DEFAULT_RETRIEVAL_TOP_K
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(RetrievalQuery(id=f"rq{len(queries) + 1}", text=text, strategy=strategy, top_k=top_k))
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
        provider=provider,
        reason=str(payload.get("reason") or "").strip()[:160],
    )
