"""问题解析辅助函数。"""

from __future__ import annotations

import re
from typing import Any

from declaration_qa.rag.retrieval_plan import infer_retrieval_plan
from declaration_qa.rag.router import ticket_context_related


def question_item(query: str, state: dict[str, Any]) -> dict[str, Any]:
    """构建子问题条目。"""
    normalized = normalize_question_text(query)
    requires_ticket_context = ticket_context_related(normalized, {**state, "query": normalized, "normalized_query": normalized})
    ticket_fact_type = normalize_ticket_fact_type("", normalized, requires_ticket_context=requires_ticket_context)
    return {
        "id": "",
        "original_query": normalized,
        "query": normalized,
        "retrieval_query": normalized,
        "requires_ticket_context": requires_ticket_context,
        "ticket_fact_type": ticket_fact_type,
        "retrieval_plan": infer_retrieval_plan(
            ticket_fact_type=ticket_fact_type,
        ),
        "is_expanded": False,
        "expansion_reason": "",
    }


def normalize_ticket_fact_type(value: str, query: str, *, requires_ticket_context: bool) -> str:
    """归一化办件事实类型。"""
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


def split_questions(query: str) -> list[str]:
    """保守拆分复合中文问题。"""
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
    """返回拆分片段是否像不同问题。"""
    signals = ["材料", "流程", "条件", "费用", "收费", "多久", "时限", "进度", "状态", "驳回", "补正", "干嘛", "是什么"]
    return sum(1 for part in parts if any(signal in part for signal in signals)) >= 2


def numbered_questions(parts: list[str]) -> list[str]:
    """清理并去重拆分后问题。"""
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
    """归一化子问题文本。"""
    return re.sub(r"\s+", " ", str(value or "").strip(" \t\r\n。；;？?"))
