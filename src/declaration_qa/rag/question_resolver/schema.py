"""问题解析数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuestionResolution:
    """问题拆解结果。"""

    is_multi_question: bool
    questions: list[dict[str, Any]]
    reason: str = ""
    provider: str = "rule"

    def payload(self) -> dict[str, Any]:
        """转换为 API 和审计载荷。"""
        return {
            "is_multi_question": self.is_multi_question,
            "questions": [dict(item) for item in self.questions],
            "reason": self.reason,
            "provider": self.provider,
        }
