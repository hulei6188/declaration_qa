"""归一化 LLM 问题解析输出。"""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.question_resolver.helpers import (
    normalize_question_text,
    normalize_ticket_fact_type,
    question_item,
)
from declaration_qa.rag.question_resolver.schema import QuestionResolution
from declaration_qa.rag.retrieval_plan import infer_retrieval_plan
from declaration_qa.rag.router import ticket_context_related


def normalize_question_resolution(payload: dict[str, Any], state: dict[str, Any]) -> QuestionResolution:
    """归一化 LLM 解析载荷。"""
    raw_questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    questions = []
    seen = set()
    for item in raw_questions:
        raw_query = item.get("query") if isinstance(item, dict) else item
        query = normalize_question_text(str(raw_query or ""))
        original_query = normalize_question_text(str(item.get("original_query") or query)) if isinstance(item, dict) else query
        if not original_query or original_query in seen:
            continue
        seen.add(original_query)
        query = original_query
        retrieval_query = normalize_question_text(str(item.get("retrieval_query") or query)) if isinstance(item, dict) else query
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
        raw_plan = item.get("retrieval_plan") if isinstance(item, dict) and isinstance(item.get("retrieval_plan"), dict) else {}
        retrieval_plan = infer_retrieval_plan(
            retrieval_query or query,
            need_complete_sections=raw_plan.get("need_complete_sections"),
            metadata_needs=raw_plan.get("metadata_needs"),
            ticket_fact_type=raw_plan.get("ticket_fact_type") or ticket_fact_type,
        )
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "original_query": original_query,
                "query": query,
                "retrieval_query": retrieval_query or query,
                "requires_ticket_context": requires_ticket_context,
                "ticket_fact_type": ticket_fact_type,
                "retrieval_plan": retrieval_plan,
                "is_expanded": False,
                "expansion_reason": "",
            }
        )
    fallback_query = str(state.get("normalized_query") or state.get("query") or "").strip()
    if not questions and fallback_query:
        questions = [question_item(fallback_query, state)]
    return QuestionResolution(
        is_multi_question=len(questions) > 1,
        questions=questions,
        reason=str(payload.get("reason") or "").strip()[:120],
        provider="llm",
    )
