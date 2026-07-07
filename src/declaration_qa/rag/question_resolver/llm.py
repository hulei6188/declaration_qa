"""LLM-based Question resolver."""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.question_resolver.normalization import normalize_question_resolution
from declaration_qa.rag.question_resolver.prompts import build_question_resolver_prompt
from declaration_qa.rag.question_resolver.rule import RuleQuestionResolver
from declaration_qa.rag.question_resolver.schema import QuestionResolution
from declaration_qa.rag.router import parse_json_object


class LLMQuestionResolver:
    """LLM-based Question resolver with rule fallback."""

    name = "llm"

    def __init__(self, llm: Any, *, fallback_resolver: RuleQuestionResolver | None = None) -> None:
        self.llm = llm
        self.fallback_resolver = fallback_resolver or RuleQuestionResolver()

    def resolve(self, state: dict[str, Any]) -> QuestionResolution:
        """Call LLM to decompose questions."""
        try:
            response = self.llm.invoke(build_question_resolver_prompt(state))
            payload = parse_json_object(str(getattr(response, "content", response)))
            resolution = normalize_question_resolution(payload, state)
            return QuestionResolution(
                is_multi_question=resolution.is_multi_question,
                questions=resolution.questions,
                reason=resolution.reason or "llm_resolver",
                provider=self.name,
            )
        except Exception as exc:
            fallback = self.fallback_resolver.resolve(state)
            return QuestionResolution(
                is_multi_question=fallback.is_multi_question,
                questions=fallback.questions,
                reason=f"llm_resolver_fallback:{type(exc).__name__}:{fallback.reason}",
                provider="rule_fallback",
            )
