"""多问题拆解 planner。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from declaration_qa.router import parse_json_object, ticket_context_related


QUESTION_PLANNER_PROMPT = """你是政务问答系统的问题拆解器，只能返回 JSON 对象，不回答业务问题。

任务:
1. 判断用户是否在一次输入中询问了多个独立问题。
2. 如果是，按原顺序拆成可以独立路由和检索的子问题。
3. 不要新增用户没问的问题，不要改写业务含义，不要输出答案。
4. 每个子问题保留必要指代词；如果用户使用“这个、该事项、当前申请”等指代，不要擅自替换成具体事项。
5. requires_ticket_context 只表示该子问题看起来是否需要当前/个人工单上下文。

JSON 字段:
{
  "is_multi_question": true,
  "questions": [
    {"query": "子问题", "requires_ticket_context": false}
  ],
  "reason": "short_english_reason"
}"""


@dataclass(frozen=True)
class QuestionPlan:
    """问题拆解结果。"""

    is_multi_question: bool
    questions: list[dict[str, Any]]
    reason: str = ""
    provider: str = "rule"

    def payload(self) -> dict[str, Any]:
        """转换为审计和响应使用的字典。"""
        return {
            "is_multi_question": self.is_multi_question,
            "questions": [dict(item) for item in self.questions],
            "reason": self.reason,
            "provider": self.provider,
        }


class RuleQuestionPlanner:
    """确定性多问题拆解器，用于本地兜底和测试。"""

    name = "rule"

    def plan(self, state: dict[str, Any]) -> QuestionPlan:
        """按常见中文连接词和问号拆分问题。"""
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


class LLMQuestionPlanner:
    """基于大模型的问题拆解器，失败时使用规则兜底。"""

    name = "llm"

    def __init__(self, llm: Any, *, fallback_planner: RuleQuestionPlanner | None = None) -> None:
        self.llm = llm
        self.fallback_planner = fallback_planner or RuleQuestionPlanner()

    def plan(self, state: dict[str, Any]) -> QuestionPlan:
        """调用大模型拆解问题。"""
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


def build_question_planner_prompt(state: dict[str, Any]) -> str:
    """构造问题拆解提示词。"""
    context = {
        "query": state.get("query") or "",
        "normalized_query": state.get("normalized_query") or "",
        "has_bound_ticket": bool(state.get("apply_id") or state.get("apply_context")),
        "current_ticket": {
            "apply_id": (state.get("apply_context") or {}).get("apply_id") or state.get("apply_id") or "",
            "biz_type": (state.get("apply_context") or {}).get("biz_type") or state.get("biz_type") or "",
            "title": (state.get("apply_context") or {}).get("title") or "",
            "status": (state.get("apply_context") or {}).get("status") or "",
            "node_name": (state.get("apply_context") or {}).get("node_name") or "",
        },
    }
    return f"{QUESTION_PLANNER_PROMPT}\n\n请求上下文 JSON:\n{json.dumps(context, ensure_ascii=False)}"


def normalize_question_plan(payload: dict[str, Any], state: dict[str, Any]) -> QuestionPlan:
    """标准化 LLM planner 输出。"""
    raw_questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    questions = []
    seen = set()
    for item in raw_questions:
        raw_query = item.get("query") if isinstance(item, dict) else item
        query = normalize_question_text(str(raw_query or ""))
        if not query or query in seen:
            continue
        seen.add(query)
        requires_ticket_context = (
            bool(item.get("requires_ticket_context"))
            if isinstance(item, dict) and "requires_ticket_context" in item
            else ticket_context_related(query, {**state, "query": query, "normalized_query": query})
        )
        questions.append({"id": f"q{len(questions) + 1}", "query": query, "requires_ticket_context": requires_ticket_context})
    fallback_query = str(state.get("normalized_query") or state.get("query") or "").strip()
    if not questions and fallback_query:
        questions = [question_item(fallback_query, state)]
    is_multi_question = len(questions) > 1
    return QuestionPlan(
        is_multi_question=is_multi_question,
        questions=questions,
        reason=str(payload.get("reason") or "").strip()[:120],
        provider="llm",
    )


def question_item(query: str, state: dict[str, Any]) -> dict[str, Any]:
    """构造子问题项。"""
    normalized = normalize_question_text(query)
    return {
        "id": "",
        "query": normalized,
        "requires_ticket_context": ticket_context_related(normalized, {**state, "query": normalized, "normalized_query": normalized}),
    }


def split_questions(query: str) -> list[str]:
    """用保守规则拆分复合问句。"""
    text = normalize_question_text(query)
    if not text:
        return []
    primary = [part for part in re.split(r"[？?；;]\s*", text) if part.strip()]
    if len(primary) > 1:
        return numbered_questions(primary)
    secondary = re.split(r"\s*(?:，|,)?(?:另外|还有|以及|并且|同时|顺便|再问下|再问一下)\s*", text)
    parts = [part for part in secondary if part.strip()]
    if len(parts) > 1 and has_distinct_question_signals(parts):
        return numbered_questions(parts)
    return [text]


def has_distinct_question_signals(parts: list[str]) -> bool:
    """判断拆分片段是否像多个独立问题。"""
    signals = ["材料", "流程", "条件", "费用", "收费", "多久", "时限", "进度", "状态", "驳回", "补正", "干嘛", "是什么"]
    return sum(1 for part in parts if any(signal in part for signal in signals)) >= 2


def numbered_questions(parts: list[str]) -> list[str]:
    """清理并去重拆分结果。"""
    output = []
    seen = set()
    for part in parts:
        item = normalize_question_text(part)
        item = re.sub(r"^(?:问题)?\s*\d+[.、:：]\s*", "", item)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def normalize_question_text(value: str) -> str:
    """清理子问题文本。"""
    return re.sub(r"\s+", " ", str(value or "").strip(" \t\r\n。；;？?"))
