"""LLM-based question planner."""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.question_planner.normalization import normalize_question_plan
from declaration_qa.rag.question_planner.prompts import build_question_planner_prompt
from declaration_qa.rag.question_planner.rule import RuleQuestionPlanner
from declaration_qa.rag.question_planner.schema import QuestionPlan
from declaration_qa.rag.router import parse_json_object


class LLMQuestionPlanner:
    """LLM-based question planner with rule fallback."""

    name = "llm"

    def __init__(self, llm: Any, *, fallback_planner: RuleQuestionPlanner | None = None) -> None:
        self.llm = llm
        self.fallback_planner = fallback_planner or RuleQuestionPlanner()

    def plan(self, state: dict[str, Any]) -> QuestionPlan:
        """Call LLM to decompose questions."""
        try:
            response = self.llm.invoke(build_question_planner_prompt(state))
            payload = parse_json_object(str(getattr(response, "content", response)))
            plan = normalize_question_plan(payload, state)
            return QuestionPlan(
                is_multi_question=plan.is_multi_question,
                questions=plan.questions,
                reason=plan.reason or "llm_planner",
                provider=self.name,
            )
        except Exception as exc:
            fallback = self.fallback_planner.plan(state)
            return QuestionPlan(
                is_multi_question=fallback.is_multi_question,
                questions=fallback.questions,
                reason=f"llm_planner_fallback:{type(exc).__name__}:{fallback.reason}",
                provider="rule_fallback",
            )
