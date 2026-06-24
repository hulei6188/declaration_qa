"""问题宏观路由、意图拆解和控制流打标 planner。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from declaration_qa.rag.router import parse_json_object, ticket_context_related


QUESTION_PLANNER_PROMPT = """你是政务问答系统的【宏观路由与意图拆解器】，只能返回 JSON 对象，不回答业务问题。

【核心任务】
1. 理解用户原始输入的复合意图。用户提问可能包含多个相对独立、需要前往不同数据源或关注不同知识维度的子问题。
2. 如果包含多意图，请按照用户的原始表达顺序，将提问拆解为可以独立处理的子问题数组（questions）。
3. 如果只有一个意图，同样返回 questions 数组，里面只放一个对象。
4. 【铁律】保持克制！直接提取用户原始片段赋予 query 字段。query 字段不负责对内容进行语义补全或改写，绝对不要擅自替换、脑补、扩展事项名称，保留一切指代词（如“这个、我这单、该事项”）。
5. retrieval_query 字段只用于 RAG 检索，可以在不改变 query 的前提下，为省略上下文的子问题补上用户原始输入、当前工单标题或请求上下文里已经明确出现的事项/证照/业务名称。

【字段打标规范】
- requires_ticket_context: 布尔值。判断该子问题是否看起来绝对依赖当前/个人的具体工单流水上下文。
  * 询问“我这单、为什么被驳回、进度到哪了、还要补什么、能不能继续办理”等个体状态 --> true
  * 询问通用的“办理条件、材料、流程、收费、时限、办事地点”等标准政策指南 --> false
- ticket_fact_type: 字符串。表示需要读取哪类工单事实。只能取: ["status", "rejection", "correction", "none"]。
- retrieval_focus: 字符串。指导下游检索的 Hint 标签。只能取: ["material_requirements", "process_steps", "eligibility", "fee_time_limit", "office_location", "contact", "online_apply", "policy_basis", "form_template", "knowledge_qa", "other"]。如果找不到合适标签，必须使用 "other"。
- retrieval_query: 字符串。下游检索使用的文本。
  * 如果子问题已经包含清楚的事项/业务上下文，retrieval_query 可以等于 query。
  * 如果子问题是“收费吗、多久、要哪些材料、去哪办”等省略问法，但同一原始问题前文已有明确事项/业务名称，应把该上下文补入 retrieval_query。
  * 只能使用请求上下文 JSON 或用户原始输入中明确出现的上下文；不能凭空补事项名称，不能把工单退回原因、办理状态当作通用政策上下文。
  * query 必须保留用户原始片段，retrieval_query 才允许为了检索而补上下文。

【拆分原则】
* 多个独立问句应拆分。
* 用“以及、还有、另外、顺便问下、再问一个”等连接的多个问题，通常应拆分。
* 同一个业务问题中的限定条件、修饰语、事项名称、地区、办理对象，不要拆分。
* 不要把“需要哪些材料”拆成多个材料项。
* 不要把“条件和材料”强行合并；如果用户同时问“条件”和“材料”，可以拆成两个子问题。
* 不要把一句话改写成更宽泛的问题。
* 用户的事实陈述、时间背景、已提交、已上传、刚办理等信息，如果没有明确询问进度、状态、结果、退回原因，不要单独生成子问题；只把它作为其他子问题的上下文。

【返回 JSON 格式】
{
  "is_multi_question": true,
  "questions": [
    {
      "original_query": "用户原始片段",
      "query": "用户原始片段",
      "retrieval_query": "用于检索的子问题文本",
      "requires_ticket_context": false,
      "ticket_fact_type": "none",
      "retrieval_focus": "retrieval_focus_tag"
    }
  ],
  "reason": "short_english_reason"
}

【标准示例】
用户输入：示例事项A办事流程，还有我那单怎么被退回了
输出：
{
  "is_multi_question": true,
  "questions": [
    {
      "original_query": "示例事项A办事流程",
      "query": "示例事项A办事流程",
      "retrieval_query": "示例事项A办事流程",
      "requires_ticket_context": false,
      "ticket_fact_type": "none",
      "retrieval_focus": "process_steps"
    },
    {
      "original_query": "还有我那单怎么被退回了",
      "query": "还有我那单怎么被退回了",
      "retrieval_query": "还有我那单怎么被退回了",
      "requires_ticket_context": true,
      "ticket_fact_type": "rejection",
      "retrieval_focus": "knowledge_qa"
    }
  ],
  "reason": "User asks for general process steps and an individual ticket rejection reason sequentially."
}

用户输入：示例事项A多久能办完？收费吗？
输出：
{
  "is_multi_question": true,
  "questions": [
    {
      "original_query": "示例事项A多久能办完",
      "query": "示例事项A多久能办完",
      "retrieval_query": "示例事项A多久能办完",
      "requires_ticket_context": false,
      "ticket_fact_type": "none",
      "retrieval_focus": "fee_time_limit"
    },
    {
      "original_query": "收费吗",
      "query": "收费吗",
      "retrieval_query": "示例事项A 收费吗",
      "requires_ticket_context": false,
      "ticket_fact_type": "none",
      "retrieval_focus": "fee_time_limit"
    }
  ],
  "reason": "The second short fee question inherits the explicit service context from the first question only for retrieval."
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
        original_query = normalize_question_text(str(item.get("original_query") or item.get("original_intent_segment") or query)) if isinstance(item, dict) else query
        if not original_query or original_query in seen:
            continue
        seen.add(original_query)
        query = original_query
        retrieval_query = normalize_question_text(
            str(item.get("retrieval_query") or item.get("search_query") or item.get("rag_query") or query)
        ) if isinstance(item, dict) else query
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
        retrieval_focus = normalize_retrieval_focus(
            str(item.get("retrieval_focus") or "") if isinstance(item, dict) else "",
            query,
        )
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "original_query": original_query,
                "query": query,
                "retrieval_query": retrieval_query or query,
                "requires_ticket_context": requires_ticket_context,
                "ticket_fact_type": ticket_fact_type,
                "retrieval_focus": retrieval_focus,
                "is_expanded": False,
                "expansion_reason": "",
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
    requires_ticket_context = ticket_context_related(normalized, {**state, "query": normalized, "normalized_query": normalized})
    return {
        "id": "",
        "original_query": normalized,
        "query": normalized,
        "retrieval_query": normalized,
        "requires_ticket_context": requires_ticket_context,
        "ticket_fact_type": normalize_ticket_fact_type("", normalized, requires_ticket_context=requires_ticket_context),
        "retrieval_focus": normalize_retrieval_focus("", normalized),
        "is_expanded": False,
        "expansion_reason": "",
    }


def normalize_ticket_fact_type(value: str, query: str, *, requires_ticket_context: bool) -> str:
    """标准化工单事实类型。"""
    allowed = {"status", "rejection", "correction", "none"}
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized if requires_ticket_context else "none"
    if not requires_ticket_context:
        return "none"
    text = str(query or "")
    if any(keyword in text for keyword in ["退回", "驳回", "不通过", "被拒", "被退", "为什么"]):
        return "rejection"
    if any(keyword in text for keyword in ["补正", "补什么", "还要补", "修改", "重新提交"]):
        return "correction"
    if any(keyword in text for keyword in ["进度", "到哪", "状态", "审核到", "办到哪", "什么情况", "怎么样"]):
        return "status"
    return "none"


def normalize_retrieval_focus(value: str, query: str) -> str:
    """标准化知识检索重点。"""
    allowed = {
        "material_requirements",
        "process_steps",
        "eligibility",
        "fee_time_limit",
        "office_location",
        "contact",
        "online_apply",
        "policy_basis",
        "form_template",
        "knowledge_qa",
        "other",
    }
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    text = str(query or "")
    if any(keyword in text for keyword in ["材料", "要带", "提交什么", "需要什么", "清单", "补什么"]):
        return "material_requirements"
    if any(keyword in text for keyword in ["流程", "步骤", "怎么办", "如何办理", "怎么申请"]):
        return "process_steps"
    if any(keyword in text for keyword in ["条件", "资格", "能不能办", "是否可以"]):
        return "eligibility"
    if any(keyword in text for keyword in ["费用", "收费", "多少钱", "时限", "多久", "几天", "多少天", "拿结果", "办结"]):
        return "fee_time_limit"
    if any(keyword in text for keyword in ["地点", "窗口", "地址", "哪里办"]):
        return "office_location"
    if any(keyword in text for keyword in ["电话", "咨询", "投诉"]):
        return "contact"
    if any(keyword in text for keyword in ["网上办", "入口", "链接", "在线"]):
        return "online_apply"
    if any(keyword in text for keyword in ["依据", "政策", "法规", "文件"]):
        return "policy_basis"
    if any(keyword in text for keyword in ["表格", "模板", "样表", "申请表"]):
        return "form_template"
    return "other"


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
