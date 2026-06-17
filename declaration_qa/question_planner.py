"""问题理解、语义补全和多问题拆解 planner。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from declaration_qa.router import parse_json_object, ticket_context_related


QUESTION_PLANNER_PROMPT = """你是政务问答系统的问题理解与拆解器，只能返回 JSON 对象，不回答业务问题。

任务：
1. 先理解用户真实问题。用户输入可能是完整问题，也可能只是搜索式关键词短语，例如“示例事项甲办事流程”“要什么证 怎么投诉”“多少天”。
2. 对关键词短语做语义补全，把它改写成适合政务知识库检索的标准、完整问题。
3. 再判断用户是否在一次输入中询问了多个相对独立、需要分别处理的问题。
4. 如果是多问题，按用户原始表达顺序拆成可以独立处理的子问题。
5. 如果不是多问题，也要返回 questions 数组，里面只放一个问题。
6. 只负责理解、补全和拆解，不判断业务意图，不输出 intent，不输出 route，不输出答案。
7. 不要新增用户没问的问题，不要补充隐藏意图，不要把背景陈述扩展成独立问题。
8. 每个子问题保留必要指代词；如果用户使用“这个、该事项、当前申请、我这单”等指代，不要擅自替换成具体事项。
9. requires_ticket_context 只表示该子问题看起来是否需要当前/个人工单上下文。
10. 用户询问“我这单、当前申请、为什么被驳回、进度到哪了、还要补什么、能不能继续办理”等，通常 requires_ticket_context=true。
11. 用户询问“办理条件、申请材料、办理流程、收费标准、办理时限、政策依据”等通用知识，通常 requires_ticket_context=false，除非明显依赖当前申请。
12. 只输出合法 JSON，不要输出 markdown，不要输出解释文字。

补全原则：
* 完整自然语言问题尽量保持原意，只做必要清理。
* 搜索式关键词短语可以补全为标准问句，例如“示例事项甲办事流程”可补全为“示例事项甲的办理流程是什么？”
* “要什么证”“需要哪些证”“要交什么材料”可补全为“办理该事项需要准备哪些证件和申请材料？”
* “多少天”“多久拿结果”可补全为“该事项的法定办结时限和承诺办结时限是多少天？”
* 如果上下文中有 current_ticket 或明确事项名称，可使用该事项名称补全短语；不要猜测无依据的具体事项。

拆分原则：
* 多个独立问句应拆分。
* 用“以及、还有、另外、顺便问下、再问一个”等连接的多个问题，通常应拆分。
* 同一个业务问题中的限定条件、修饰语、事项名称、地区、办理对象，不要拆分。
* 不要把“需要哪些材料”拆成多个材料项。
* 不要把“条件和材料”强行合并；如果用户同时问“条件”和“材料”，可以拆成两个子问题。
* 不要把一句话改写成更宽泛的问题。
* 用户的事实陈述、时间背景、已提交、已上传、刚办理等信息，如果没有明确询问进度、状态、结果、退回原因，不要单独生成子问题；只把它作为其他子问题的上下文。

返回 JSON 格式：
{
  "is_multi_question": true,
  "questions": [
    {
      "original_query": "用户原始片段",
      "query": "理解和必要补全后的标准独立问题",
      "requires_ticket_context": false,
      "is_expanded": false,
      "expansion_reason": ""
    }
  ],
  "reason": "short_english_reason"
}

示例 1：
用户：示例事项甲办事流程，还有要交什么材料
输出：
{
  "is_multi_question": true,
  "questions": [
    {
      "original_query": "示例事项甲办事流程",
      "query": "示例事项甲的办理流程是什么？",
      "requires_ticket_context": false,
      "is_expanded": true,
      "expansion_reason": "keyword_query_to_process_question"
    },
    {
      "original_query": "还有要交什么材料",
      "query": "示例事项甲需要提交哪些申请材料？",
      "requires_ticket_context": false,
      "is_expanded": true,
      "expansion_reason": "keyword_query_to_material_question"
    }
  ],
  "reason": "User asks two searchable knowledge questions."
}

示例 2：
用户：我这单为什么被驳回，还要补什么材料？
输出：
{
  "is_multi_question": true,
  "questions": [
    {
      "original_query": "我这单为什么被驳回",
      "query": "我这单为什么被驳回？",
      "requires_ticket_context": true,
      "is_expanded": false,
      "expansion_reason": ""
    },
    {
      "original_query": "还要补什么材料",
      "query": "还要补什么材料？",
      "requires_ticket_context": true,
      "is_expanded": false,
      "expansion_reason": ""
    }
  ],
  "reason": "User asks two ticket-related questions."
}

示例 3：
用户：办理示例证照乙需要哪些材料？
输出：
{
  "is_multi_question": false,
  "questions": [
    {
      "original_query": "办理示例证照乙需要哪些材料？",
      "query": "办理示例证照乙需要哪些材料？",
      "requires_ticket_context": false,
      "is_expanded": false,
      "expansion_reason": ""
    }
  ],
  "reason": "User asks one knowledge question."
}

示例 4：
用户：我今天材料交上去了，最快几天能拿到结果？
输出：
{
  "is_multi_question": false,
  "questions": [
    {
      "original_query": "最快几天能拿到结果",
      "query": "该事项提交申请后最快几天能拿到审批结果，法定办结时限和承诺办结时限是多少天？",
      "requires_ticket_context": false,
      "is_expanded": true,
      "expansion_reason": "background_statement_plus_time_limit_question"
    }
  ],
  "reason": "Submission statement is background, user asks one time-limit question."
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
    apply_context = state.get("apply_context") or {}
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    current_ticket_title = (
        apply_context.get("title")
        or apply_context.get("item_name")
        or payload.get("official_item_name")
        or payload.get("item_name")
        or payload.get("title")
        or payload.get("catalog_name")
        or ""
    )
    context = {
        "query": state.get("query") or "",
        "normalized_query": state.get("normalized_query") or "",
        "has_bound_ticket": bool(state.get("apply_id") or state.get("apply_context")),
        "current_ticket": {
            "apply_id": apply_context.get("apply_id") or state.get("apply_id") or "",
            "biz_type": apply_context.get("biz_type") or state.get("biz_type") or "",
            "title": current_ticket_title,
            "status": apply_context.get("status") or "",
            "node_name": apply_context.get("node_name") or "",
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
        original_query = normalize_question_text(str(item.get("original_query") or item.get("original_intent_segment") or query)) if isinstance(item, dict) else query
        requires_ticket_context = (
            bool(item.get("requires_ticket_context"))
            if isinstance(item, dict) and "requires_ticket_context" in item
            else ticket_context_related(query, {**state, "query": query, "normalized_query": query})
        )
        is_expanded = bool(item.get("is_expanded")) if isinstance(item, dict) else original_query != query
        expansion_reason = str(item.get("expansion_reason") or "").strip()[:120] if isinstance(item, dict) else ""
        if requires_ticket_context and original_query:
            query = original_query
            is_expanded = False
            expansion_reason = "ticket_related_query_kept_unexpanded"
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "original_query": original_query,
                "query": query,
                "requires_ticket_context": requires_ticket_context,
                "is_expanded": is_expanded,
                "expansion_reason": expansion_reason,
            }
        )
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
        "original_query": normalized,
        "query": normalized,
        "requires_ticket_context": ticket_context_related(normalized, {**state, "query": normalized, "normalized_query": normalized}),
        "is_expanded": False,
        "expansion_reason": "",
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
