"""Normalize LLM question planner outputs."""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.question_planner.helpers import (
    normalize_question_text,
    normalize_retrieval_focus,
    normalize_ticket_fact_type,
    question_item,
)
from declaration_qa.rag.question_planner.schema import QuestionPlan
from declaration_qa.rag.router import ticket_context_related


def normalize_question_plan(payload: dict[str, Any], state: dict[str, Any]) -> QuestionPlan:
    """Normalize LLM planner payload."""
    raw_questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    questions = []
    seen = set()
    for item in raw_questions:
        raw_query = item.get("query") if isinstance(item, dict) else item
        query = normalize_question_text(str(raw_query or ""))
        original_query = (
            normalize_question_text(str(item.get("original_query") or item.get("original_intent_segment") or query))
            if isinstance(item, dict)
            else query
        )
        if not original_query or original_query in seen:
            continue
        seen.add(original_query)
        query = original_query
        retrieval_query = (
            normalize_question_text(str(item.get("retrieval_query") or item.get("search_query") or item.get("rag_query") or query))
            if isinstance(item, dict)
            else query
        )
        requires_ticket_context = (
            bool(item.get("requires_ticket_context"))
            if isinstance(item, dict) and "requires_ticket_context" in item
            else ticket_context_related(query, {**state, "query": query, "normalized_query": query})
        )
        ticket_fact_type = normalize_ticket_fact_type(
            str(item.get("ticket_fact_type") or "") if isinstance(item, dict) else "",
            query,
            requires_ticket_context=requires_ticket_context,
        )
        retrieval_focus = normalize_retrieval_focus(
            str(item.get("retrieval_focus") or "") if isinstance(item, dict) else "",
            query,
        )
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "original_query": original_query,
                "query": query,
                "retrieval_query": retrieval_query or query,
                "requires_ticket_context": requires_ticket_context,
                "ticket_fact_type": ticket_fact_type,
                "retrieval_focus": retrieval_focus,
                "is_expanded": False,
                "expansion_reason": "",
            }
        )
    fallback_query = str(state.get("normalized_query") or state.get("query") or "").strip()
    if not questions and fallback_query:
        questions = [question_item(fallback_query, state)]
    return QuestionPlan(
        is_multi_question=len(questions) > 1,
        questions=questions,
        reason=str(payload.get("reason") or "").strip()[:120],
        provider="llm",
    )
