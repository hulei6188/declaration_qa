"""Rule-based question planner."""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.question_planner.helpers import question_item, split_questions
from declaration_qa.rag.question_planner.schema import QuestionPlan


class RuleQuestionPlanner:
    """Deterministic multi-question planner for local fallback and tests."""

    name = "rule"

    def plan(self, state: dict[str, Any]) -> QuestionPlan:
        """Split questions with conservative Chinese rules."""
        query = str(state.get("normalized_query") or state.get("query") or "").strip()
        parts = split_questions(query)
        if len(parts) <= 1:
            return QuestionPlan(
                is_multi_question=False,
                questions=[question_item(query, state)] if query else [],
                reason="single_question",
                provider=self.name,
            )
        return QuestionPlan(
            is_multi_question=True,
            questions=[question_item(part, state) for part in parts],
            reason="rule_split",
            provider=self.name,
        )
