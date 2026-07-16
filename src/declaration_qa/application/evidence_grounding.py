"""最终回答的确定性证据约束。"""

from __future__ import annotations

import re
from typing import Any

from declaration_qa.application.evidence_registry import (
    build_evidence_registry,
    registered_business_results,
    registered_knowledge_sources,
)
from declaration_qa.rag.evidence import source_traceable
from declaration_qa.rag.prompting import source_evidence_text


KNOWLEDGE_MARKER_PATTERN = re.compile(r"\[证据(\d+)\]")
BUSINESS_MARKER_PATTERN = re.compile(r"\[业务(\d+)\]")
LEGAL_REFERENCE_PATTERN = re.compile(r"(?:《[^》\n]{2,}》\s*)?第[〇零一二三四五六七八九十百千\d]+条")
BUSINESS_IDENTIFIER_PATTERNS: dict[str, re.Pattern[str]] = {
    "application_id": re.compile(r"(?:application_id|申请编号|申请ID)\s*[:：#]?\s*(\d+)", re.IGNORECASE),
    "appointment_id": re.compile(r"(?:appointment_id|预约编号|预约ID)\s*[:：#]?\s*(\d+)", re.IGNORECASE),
    "payment_order_id": re.compile(r"(?:payment_order_id|缴费订单编号|缴费订单ID)\s*[:：#]?\s*(\d+)", re.IGNORECASE),
    "delivery_id": re.compile(r"(?:delivery_id|送达编号|送达ID)\s*[:：#]?\s*(\d+)", re.IGNORECASE),
}

KNOWLEDGE_QUERY_TERMS = (
    "材料",
    "政策",
    "法规",
    "条例",
    "办法",
    "规定",
    "依据",
    "流程",
    "条件",
    "费用",
    "收费",
    "时限",
    "期限",
    "模板",
    "样表",
    "协议",
    "证明",
    "法定",
)
KNOWLEDGE_QUERY_PATTERNS = (
    "怎么办理",
    "如何办理",
    "办理什么",
    "办理流程",
    "办理条件",
    "办理材料",
    "办理时限",
    "办理依据",
)
BUSINESS_QUERY_TERMS = (
    "我的申请",
    "申请进度",
    "申请状态",
    "退回原因",
    "预约",
    "排队",
    "缴费",
    "票据",
    "回执",
    "办结结果",
    "送达",
    "咨询记录",
)
def validate_grounded_answer(state: dict[str, Any], answer: str) -> dict[str, Any]:
    """只检查回答中显式引用的句子及其底层证据。"""
    query = str(state.get("normalized_query") or state.get("query") or "").strip()
    sources = [item for item in (state.get("retrieved_sources") or []) if isinstance(item, dict)]
    registered_sources = registered_knowledge_sources(sources)
    registered_business = registered_business_results(state.get("tool_results") or [])
    business_results = [result for _entry, result in registered_business]
    evidence_registry = build_evidence_registry(sources, state.get("tool_results") or [])
    valid_source_indexes = {
        index
        for index, (_entry, source) in enumerate(registered_sources, start=1)
        if source_traceable(source) and bool(normalize_evidence_text(source_evidence_text(source)))
    }
    valid_business_indexes = set(range(1, len(registered_business) + 1))
    requires_knowledge = query_requires_knowledge(query)
    requires_business = query_requires_business(query)
    has_grounding_flow = bool(sources or business_results or state.get("tool_results"))

    checks: dict[str, Any] = {
        "knowledge_source_count": len(registered_sources),
        "traceable_knowledge_source_count": len(valid_source_indexes),
        "business_evidence_count": len(business_results),
        "requires_knowledge": requires_knowledge,
        "requires_business": requires_business,
        "evidence_registry": evidence_registry,
    }
    if not has_grounding_flow:
        if requires_knowledge or requires_business:
            return grounding_decision(False, "no_trusted_evidence", "grounded", checks)
        return grounding_decision(True, "", "direct", checks)

    if requires_knowledge and not valid_source_indexes:
        return grounding_decision(False, "no_traceable_knowledge_evidence", "grounded", checks)
    if requires_business and not valid_business_indexes:
        return grounding_decision(False, "no_successful_business_evidence", "grounded", checks)

    knowledge_citations = {int(item) for item in KNOWLEDGE_MARKER_PATTERN.findall(answer)}
    business_citations = {int(item) for item in BUSINESS_MARKER_PATTERN.findall(answer)}
    invalid_knowledge = sorted(knowledge_citations - valid_source_indexes)
    invalid_business = sorted(business_citations - valid_business_indexes)
    checks.update(
        {
            "knowledge_citations": sorted(knowledge_citations),
            "business_citations": sorted(business_citations),
            "invalid_knowledge_citations": invalid_knowledge,
            "invalid_business_citations": invalid_business,
        }
    )
    if invalid_knowledge or invalid_business:
        return grounding_decision(False, "invalid_evidence_citation", "grounded", checks)

    business_identity_mismatches: list[dict[str, Any]] = []
    for unit in cited_units(answer):
        cited_business = {int(item) for item in BUSINESS_MARKER_PATTERN.findall(unit)}
        mismatch = business_identity_mismatch(unit, cited_business, registered_business)
        if mismatch:
            business_identity_mismatches.append(mismatch)
    checks["business_identity_mismatches"] = business_identity_mismatches[:10]
    if business_identity_mismatches:
        return grounding_decision(False, "business_key_mismatch", "grounded", checks)

    return grounding_decision(True, "", "grounded", checks)


def successful_business_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """返回已绑定调用标识和业务主键的成功业务结果。"""
    return [result for _entry, result in registered_business_results(results)]


def business_identity_mismatch(
    unit: str,
    cited_indexes: set[int],
    registered_business: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any] | None:
    """检查事实行中的显式业务标识是否属于所引业务证据。"""
    if not cited_indexes:
        return None
    cited_entries = [
        entry
        for index, (entry, _result) in enumerate(registered_business, start=1)
        if index in cited_indexes
    ]
    for field, pattern in BUSINESS_IDENTIFIER_PATTERNS.items():
        mentioned = {int(value) for value in pattern.findall(unit)}
        if not mentioned:
            continue
        allowed = {
            int(value)
            for entry in cited_entries
            for value in dict(entry.get("business_key") or {}).get(field) or []
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        }
        if allowed and not mentioned.issubset(allowed):
            return {
                "field": field,
                "mentioned": sorted(mentioned),
                "allowed": sorted(allowed),
                "unit": short_check_text(unit),
            }
    return None


def query_requires_knowledge(query: str) -> bool:
    """判断问题是否要求外部公开知识依据。"""
    return (
        bool(LEGAL_REFERENCE_PATTERN.search(query))
        or any(term in query for term in KNOWLEDGE_QUERY_TERMS)
        or any(pattern in query for pattern in KNOWLEDGE_QUERY_PATTERNS)
    )


def query_requires_business(query: str) -> bool:
    """判断问题是否要求当前用户的业务工具事实。"""
    return any(term in query for term in BUSINESS_QUERY_TERMS)


def cited_units(answer: str) -> list[str]:
    """只提取包含知识或业务引用标记的文本行。"""
    return [
        line
        for raw_line in str(answer or "").splitlines()
        if (line := raw_line.strip())
        and (
            KNOWLEDGE_MARKER_PATTERN.search(line)
            or BUSINESS_MARKER_PATTERN.search(line)
        )
    ]


def normalize_evidence_text(value: str) -> str:
    """仅消除排版空白，保留原文用词和标点用于严格匹配。"""
    return re.sub(r"\s+", "", str(value or ""))


def short_check_text(value: str, limit: int = 160) -> str:
    """限制审计日志中的失败文本长度。"""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def grounding_decision(ok: bool, reason: str, mode: str, checks: dict[str, Any]) -> dict[str, Any]:
    """构造统一的证据闸门结果。"""
    return {
        "ok": bool(ok),
        "reason": str(reason or ""),
        "mode": mode,
        "evidence_registry": checks.get("evidence_registry") or {"knowledge": [], "business": []},
        "checks": checks,
    }


def isolate_untrusted_assistant_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """兼容旧调用方，返回仅供指代理解的历史对话投影。"""
    from declaration_qa.memory.evidence import project_historical_messages

    return project_historical_messages(messages)
