"""基于规则的问题解析器。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.rag.question_resolver.helpers import question_item, split_questions
from gov_service_agent.rag.question_resolver.schema import QuestionResolution


class RuleQuestionResolver:
    """用于本地兜底和测试的确定性多问题解析器。"""

    name = "rule"

    def resolve(self, state: dict[str, Any]) -> QuestionResolution:
        """使用保守中文规则拆分问题。"""
        query = str(state.get("normalized_query") or state.get("query") or "").strip()
        parts = split_questions(query)
        if len(parts) <= 1:
            return QuestionResolution(
                is_multi_question=False,
                questions=[question_item(query, state)] if query else [],
                reason="single_question",
                provider=self.name,
            )
        return QuestionResolution(
            is_multi_question=True,
            questions=[question_item(part, state) for part in parts],
            reason="rule_split",
            provider=self.name,
        )
