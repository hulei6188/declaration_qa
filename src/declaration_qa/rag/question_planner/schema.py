"""Question planner data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuestionPlan:
    """Question decomposition result."""

    is_multi_question: bool
    questions: list[dict[str, Any]]
    reason: str = ""
    provider: str = "rule"

    def payload(self) -> dict[str, Any]:
        """Convert to API/audit payload."""
        return {
            "is_multi_question": self.is_multi_question,
            "questions": [dict(item) for item in self.questions],
            "reason": self.reason,
            "provider": self.provider,
        }
