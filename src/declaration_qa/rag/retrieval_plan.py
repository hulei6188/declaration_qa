"""结构化检索计划辅助函数。"""

from __future__ import annotations

from typing import Any

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
    "申请书",
    "申请表",
    "表格",
    "空表",
    "样表",
    "样本",
    "模板",
    "怎么填",
    "如何填",
    "填写",
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
    *,
    need_complete_sections: Any = None,
    metadata_needs: Any = None,
    ticket_fact_type: str = "none",
) -> dict[str, Any]:
    """推断并归一化结构化检索计划。"""
    query = str(base_query or "")
    sections = list(normalize_complete_sections(need_complete_sections))

    if contains_any(query, BROAD_HANDLING_KEYWORDS):
        append_unique(sections, "申请材料")
        append_unique(sections, "办理流程")
    if contains_any(query, MATERIAL_KEYWORDS):
        append_unique(sections, "申请材料")
    if contains_any(query, PROCESS_KEYWORDS):
        append_unique(sections, "办理流程")

    return {
        "need_complete_sections": sections,
        "metadata_needs": list(normalize_metadata_needs(metadata_needs)),
        "ticket_fact_type": normalize_ticket_fact_type(ticket_fact_type),
    }


def normalize_retrieval_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """归一化已经生成的检索计划。"""
    raw = plan or {}
    return {
        "need_complete_sections": list(normalize_complete_sections(raw.get("need_complete_sections"))),
        "metadata_needs": list(normalize_metadata_needs(raw.get("metadata_needs"))),
        "ticket_fact_type": normalize_ticket_fact_type(str(raw.get("ticket_fact_type") or "none")),
    }


def backfill_chunk_types_from_plan(plan: dict[str, Any] | None) -> list[str]:
    """返回完整章节计划请求的结构化片段类型。"""
    normalized = normalize_retrieval_plan(plan)
    output: list[str] = []
    for section in normalized["need_complete_sections"]:
        append_unique(output, CHUNK_TYPE_BY_COMPLETE_SECTION.get(section, ""))
    return output


def normalize_complete_sections(values: Any) -> tuple[str, ...]:
    """归一化完整章节取值。"""
    output: list[str] = []
    for value in list_values(values):
        key = str(value or "").strip()
        normalized = COMPLETE_SECTION_ALIASES.get(key, COMPLETE_SECTION_ALIASES.get(key.lower(), key))
        if normalized in CHUNK_TYPE_BY_COMPLETE_SECTION:
            append_unique(output, normalized)
    return tuple(output)


def normalize_metadata_needs(values: Any) -> tuple[str, ...]:
    """归一化元数据需求取值。"""
    output: list[str] = []
    for value in list_values(values):
        normalized = str(value or "").strip().lower()
        if normalized in ALLOWED_METADATA_NEEDS:
            append_unique(output, normalized)
    return tuple(output)


def normalize_ticket_fact_type(value: str) -> str:
    """归一化检索计划中的办件事实类型。"""
    normalized = str(value or "none").strip().lower()
    return normalized if normalized in {"status", "rejection", "correction", "none"} else "none"


def list_values(values: Any) -> list[Any]:
    """将类列表值返回为扁平列表。"""
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """返回文本是否包含任一关键词。"""
    return any(keyword in text for keyword in keywords)


def append_unique(output: list[str], value: str) -> None:
    """仅追加一次非空值。"""
    if value and value not in output:
        output.append(value)
