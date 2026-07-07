"""Structured retrieval plan helpers."""

from __future__ import annotations

from typing import Any


ALLOWED_EVIDENCE_TYPES = (
    "basic_info",
    "eligibility",
    "material_item",
    "process_step",
    "fee_time_limit",
    "office_location",
    "contact",
    "policy_basis",
    "form_template",
    "online_apply",
)

EVIDENCE_TYPE_ALIASES = {
    "basic": "basic_info",
    "basic_information": "basic_info",
    "条件": "eligibility",
    "办理条件": "eligibility",
    "受理条件": "eligibility",
    "材料": "material_item",
    "申请材料": "material_item",
    "materials": "material_item",
    "material": "material_item",
    "流程": "process_step",
    "办理流程": "process_step",
    "process": "process_step",
    "steps": "process_step",
    "时限": "fee_time_limit",
    "办理时限": "fee_time_limit",
    "地点": "office_location",
    "办理地点": "office_location",
    "电话": "contact",
    "联系方式": "contact",
    "依据": "policy_basis",
    "法律依据": "policy_basis",
    "表格": "form_template",
    "申请表": "form_template",
    "网办": "online_apply",
    "网上办理": "online_apply",
}

FOCUS_EVIDENCE_TYPES = {
    "material_requirements": ("material_item",),
    "process_steps": ("process_step",),
    "eligibility": ("eligibility",),
    "fee_time_limit": ("fee_time_limit",),
    "office_location": ("office_location",),
    "contact": ("contact",),
    "online_apply": ("online_apply",),
    "policy_basis": ("policy_basis",),
    "form_template": ("form_template", "material_item"),
    "knowledge_qa": ("basic_info",),
}

COMPLETE_SECTION_ALIASES = {
    "材料": "申请材料",
    "申请材料": "申请材料",
    "材料清单": "申请材料",
    "materials": "申请材料",
    "流程": "办理流程",
    "办理流程": "办理流程",
    "办理环节": "办理流程",
    "process": "办理流程",
}

ALLOWED_METADATA_NEEDS = (
    "region_code",
    "policy_version",
    "item_scope",
    "ticket_scope",
    "as_of",
)

CHUNK_TYPE_BY_COMPLETE_SECTION = {
    "申请材料": "material_item",
    "办理流程": "process_step",
}

BROAD_HANDLING_KEYWORDS = (
    "需要办理什么",
    "要办理什么",
    "该办什么",
    "办什么",
    "怎么办理",
    "怎么办",
    "怎么弄",
    "如何办理",
)

MATERIAL_KEYWORDS = (
    "材料",
    "申请材料",
    "要交啥",
    "要准备什么",
    "提交什么",
    "上传什么",
    "需要哪些材料",
    "要带",
    "需要什么材料",
)

PROCESS_KEYWORDS = (
    "流程",
    "办理步骤",
    "办理环节",
    "提交后",
    "审核",
    "怎么申请",
    "如何办理",
)


def infer_retrieval_plan(
    base_query: str,
    retrieval_focus: str,
    *,
    evidence_types: Any = None,
    need_complete_sections: Any = None,
    metadata_needs: Any = None,
    ticket_fact_type: str = "none",
) -> dict[str, Any]:
    """Infer a normalized structured retrieval plan."""
    query = str(base_query or "")
    focus = str(retrieval_focus or "")
    inferred_types = list(normalize_evidence_types(evidence_types))
    for evidence_type in FOCUS_EVIDENCE_TYPES.get(focus, ()):
        append_unique(inferred_types, evidence_type)

    if contains_any(query, BROAD_HANDLING_KEYWORDS):
        for evidence_type in ("basic_info", "eligibility", "material_item", "process_step"):
            append_unique(inferred_types, evidence_type)
    if contains_any(query, MATERIAL_KEYWORDS):
        append_unique(inferred_types, "material_item")
    if contains_any(query, PROCESS_KEYWORDS):
        append_unique(inferred_types, "process_step")

    sections = list(normalize_complete_sections(need_complete_sections))
    if "material_item" in inferred_types and (
        focus == "material_requirements"
        or contains_any(query, MATERIAL_KEYWORDS)
        or contains_any(query, BROAD_HANDLING_KEYWORDS)
    ):
        append_unique(sections, "申请材料")
    if "process_step" in inferred_types and (
        focus == "process_steps"
        or contains_any(query, PROCESS_KEYWORDS)
        or contains_any(query, BROAD_HANDLING_KEYWORDS)
    ):
        append_unique(sections, "办理流程")

    return {
        "evidence_types": inferred_types,
        "need_complete_sections": sections,
        "metadata_needs": list(normalize_metadata_needs(metadata_needs)),
        "ticket_fact_type": normalize_ticket_fact_type(ticket_fact_type),
    }


def normalize_retrieval_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize an already produced retrieval plan."""
    raw = plan or {}
    return {
        "evidence_types": list(normalize_evidence_types(raw.get("evidence_types"))),
        "need_complete_sections": list(normalize_complete_sections(raw.get("need_complete_sections"))),
        "metadata_needs": list(normalize_metadata_needs(raw.get("metadata_needs"))),
        "ticket_fact_type": normalize_ticket_fact_type(str(raw.get("ticket_fact_type") or "none")),
    }


def backfill_chunk_types_from_plan(plan: dict[str, Any] | None) -> list[str]:
    """Return structured chunk types requested by complete-section plan."""
    normalized = normalize_retrieval_plan(plan)
    output: list[str] = []
    for section in normalized["need_complete_sections"]:
        append_unique(output, CHUNK_TYPE_BY_COMPLETE_SECTION.get(section, ""))
    return output


def normalize_evidence_types(values: Any) -> tuple[str, ...]:
    """Normalize evidence type values."""
    output: list[str] = []
    for value in list_values(values):
        key = str(value or "").strip()
        normalized = EVIDENCE_TYPE_ALIASES.get(key, EVIDENCE_TYPE_ALIASES.get(key.lower(), key.lower()))
        if normalized in ALLOWED_EVIDENCE_TYPES:
            append_unique(output, normalized)
    return tuple(output)


def normalize_complete_sections(values: Any) -> tuple[str, ...]:
    """Normalize complete-section values."""
    output: list[str] = []
    for value in list_values(values):
        key = str(value or "").strip()
        normalized = COMPLETE_SECTION_ALIASES.get(key, COMPLETE_SECTION_ALIASES.get(key.lower(), key))
        if normalized in CHUNK_TYPE_BY_COMPLETE_SECTION:
            append_unique(output, normalized)
    return tuple(output)


def normalize_metadata_needs(values: Any) -> tuple[str, ...]:
    """Normalize metadata need values."""
    output: list[str] = []
    for value in list_values(values):
        normalized = str(value or "").strip().lower()
        if normalized in ALLOWED_METADATA_NEEDS:
            append_unique(output, normalized)
    return tuple(output)


def normalize_ticket_fact_type(value: str) -> str:
    """Normalize ticket fact type inside retrieval plan."""
    normalized = str(value or "none").strip().lower()
    return normalized if normalized in {"status", "rejection", "correction", "none"} else "none"


def list_values(values: Any) -> list[Any]:
    """Return list-like values as a flat list."""
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether text contains any keyword."""
    return any(keyword in text for keyword in keywords)


def append_unique(output: list[str], value: str) -> None:
    """Append non-empty value once."""
    if value and value not in output:
        output.append(value)
