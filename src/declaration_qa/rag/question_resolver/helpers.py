"""Question resolver helper functions."""

from __future__ import annotations

import re
from typing import Any

from declaration_qa.rag.retrieval_plan import infer_retrieval_plan
from declaration_qa.rag.router import ticket_context_related


def question_item(query: str, state: dict[str, Any]) -> dict[str, Any]:
    """Build a sub-question item."""
    normalized = normalize_question_text(query)
    requires_ticket_context = ticket_context_related(normalized, {**state, "query": normalized, "normalized_query": normalized})
    ticket_fact_type = normalize_ticket_fact_type("", normalized, requires_ticket_context=requires_ticket_context)
    retrieval_focus = normalize_retrieval_focus("", normalized)
    return {
        "id": "",
        "original_query": normalized,
        "query": normalized,
        "retrieval_query": normalized,
        "requires_ticket_context": requires_ticket_context,
        "ticket_fact_type": ticket_fact_type,
        "retrieval_focus": retrieval_focus,
        "retrieval_plan": infer_retrieval_plan(
            normalized,
            retrieval_focus,
            ticket_fact_type=ticket_fact_type,
        ),
        "is_expanded": False,
        "expansion_reason": "",
    }


def normalize_ticket_fact_type(value: str, query: str, *, requires_ticket_context: bool) -> str:
    """Normalize ticket fact type."""
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
    """Normalize knowledge retrieval focus."""
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
    """Split a compound Chinese question conservatively."""
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
    """Return whether split parts look like distinct questions."""
    signals = ["材料", "流程", "条件", "费用", "收费", "多久", "时限", "进度", "状态", "驳回", "补正", "干嘛", "是什么"]
    return sum(1 for part in parts if any(signal in part for signal in signals)) >= 2


def numbered_questions(parts: list[str]) -> list[str]:
    """Clean and deduplicate split questions."""
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
    """Normalize sub-question text."""
    return re.sub(r"\s+", " ", str(value or "").strip(" \t\r\n。；;？?"))
