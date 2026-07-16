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
KNOWLEDGE_LINE_TERMS = (
    "政策",
    "法规",
    "条例",
    "办法",
    "规定",
    "法定",
    "材料",
    "办理流程",
    "办理条件",
    "费用",
    "收费",
    "时限",
    "期限",
    "模板",
    "样表",
    "协议",
    "证明",
    "应当",
    "必须",
    "不得",
)
BUSINESS_LINE_TERMS = (
    "application_id",
    "申请编号",
    "当前状态",
    "提交时间",
    "办结时间",
    "退回",
    "补正",
    "预约号",
    "排队号",
    "等待人数",
    "等待时间",
    "窗口",
    "已签到",
    "缴费",
    "票据",
    "回执",
    "送达",
)
UNCERTAINTY_PREFIXES = (
    "现有证据无法确认",
    "当前证据无法确认",
    "现有公开材料未说明",
    "当前公开材料未说明",
    "暂未查询到",
    "建议联系",
    "建议向",
    "请联系",
    "请向",
)


def validate_grounded_answer(state: dict[str, Any], answer: str) -> dict[str, Any]:
    """检查最终回答是否绑定了存在且可追溯的本轮证据。"""
    query = str(state.get("normalized_query") or state.get("query") or "").strip()
    sources = [item for item in (state.get("retrieved_sources") or []) if isinstance(item, dict)]
    registered_sources = registered_knowledge_sources(sources)
    registered_business = registered_business_results(state.get("tool_results") or [])
    grounded_sources = [source for _entry, source in registered_sources]
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
    for unit in factual_units(answer):
        cited_business = {int(item) for item in BUSINESS_MARKER_PATTERN.findall(unit)}
        mismatch = business_identity_mismatch(unit, cited_business, registered_business)
        if mismatch:
            business_identity_mismatches.append(mismatch)
    checks["business_identity_mismatches"] = business_identity_mismatches[:10]
    if business_identity_mismatches:
        return grounding_decision(False, "business_key_mismatch", "grounded", checks)

    historical_indexes = {
        index
        for index, (_entry, result) in enumerate(registered_business, start=1)
        if str(result.get("memory_origin") or "") == "historical_snapshot"
    }
    cited_historical = historical_indexes & business_citations
    checks["historical_business_citations"] = sorted(cited_historical)
    if cited_historical and not answer_has_historical_time_context(answer):
        return grounding_decision(False, "historical_snapshot_without_time_context", "grounded", checks)

    uncited_units: list[str] = []
    for unit in factual_units(answer):
        cited_knowledge = {int(item) for item in KNOWLEDGE_MARKER_PATTERN.findall(unit)}
        cited_business = {int(item) for item in BUSINESS_MARKER_PATTERN.findall(unit)}
        needs_knowledge = line_requires_knowledge(unit, grounded_sources, business_results)
        needs_business = line_requires_business(unit, grounded_sources, business_results)
        if needs_knowledge and not (cited_knowledge & valid_source_indexes):
            uncited_units.append(short_check_text(unit))
            continue
        if needs_business and not (cited_business & valid_business_indexes):
            uncited_units.append(short_check_text(unit))
            continue
        if not needs_knowledge and not needs_business and not (
            (cited_knowledge & valid_source_indexes) or (cited_business & valid_business_indexes)
        ):
            uncited_units.append(short_check_text(unit))
            continue

    checks["uncited_fact_units"] = uncited_units[:10]
    if uncited_units:
        return grounding_decision(False, "fact_without_evidence_citation", "grounded", checks)
    if not knowledge_citations and not business_citations:
        return grounding_decision(False, "answer_has_no_evidence_citation", "grounded", checks)
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


def answer_has_historical_time_context(answer: str) -> bool:
    """判断历史业务快照回答是否明确表达观察时点。"""
    text = str(answer or "")
    if any(term in text for term in ("截至", "查询时间", "观察时间", "当时", "上一轮", "刚才", "先前")):
        return True
    return bool(re.search(r"20\d{2}[-年/]\d{1,2}(?:[-月/]\d{1,2})?", text))


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


def factual_units(answer: str) -> list[str]:
    """提取需要逐条绑定引用的事实行。"""
    output: list[str] = []
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if not line or line in {"---", "***"}:
            continue
        if line.startswith("#"):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        plain = re.sub(r"^[\-*+>\s\d.、）)]+", "", line).strip()
        if not plain or is_heading(plain) or plain.startswith(UNCERTAINTY_PREFIXES):
            continue
        if is_factual_line(plain):
            output.append(line)
    return output


def is_heading(line: str) -> bool:
    """识别不承载事实结论的短标题。"""
    return len(line) <= 24 and line.endswith(("：", ":")) and not re.search(r"[。；]", line)


def is_factual_line(line: str) -> bool:
    """将证据回答中的完整文本行视为待核验事实单元。"""
    without_markers = KNOWLEDGE_MARKER_PATTERN.sub("", BUSINESS_MARKER_PATTERN.sub("", line)).strip()
    if len(without_markers) < 4:
        return False
    return True


def line_requires_knowledge(
    line: str,
    sources: list[dict[str, Any]],
    business_results: list[dict[str, Any]],
) -> bool:
    """判断事实行是否必须引用公开知识证据。"""
    if LEGAL_REFERENCE_PATTERN.search(line):
        return True
    if sources and any(term in line for term in KNOWLEDGE_LINE_TERMS):
        return True
    return bool(sources) and not business_results


def line_requires_business(
    line: str,
    sources: list[dict[str, Any]],
    business_results: list[dict[str, Any]],
) -> bool:
    """判断事实行是否必须引用个人业务工具结果。"""
    if business_results and any(term in line for term in BUSINESS_LINE_TERMS):
        return True
    return bool(business_results) and not sources


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
