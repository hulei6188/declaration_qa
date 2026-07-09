"""resolver 结果写回状态的辅助函数。"""

from __future__ import annotations

from typing import Any

from declaration_qa.application.workflow_helpers.intent import infer_biz_type
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.retrieval_plan import normalize_retrieval_plan


def apply_planned_question(state: dict[str, Any], question: dict[str, Any]) -> None:
    """将 resolver 理解后的单问题写回状态，保留原始问题在 question_resolution 中。"""
    planned_query = normalize_text(str(question.get("original_query") or question.get("query") or ""))
    if not planned_query:
        return
    state["query"] = planned_query
    retrieval_query = normalize_text(str(question.get("retrieval_query") or planned_query))
    state["normalized_query"] = retrieval_query
    state["retrieval_query"] = retrieval_query
    state["requires_ticket_context"] = bool(question.get("requires_ticket_context"))
    state["ticket_fact_type"] = question.get("ticket_fact_type") or "none"
    state["retrieval_plan"] = normalize_retrieval_plan(question.get("retrieval_plan") or {})
    state["biz_type"] = state.get("biz_type") or infer_biz_type(retrieval_query) or infer_biz_type(planned_query)
