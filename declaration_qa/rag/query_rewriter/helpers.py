"""Helper functions for query rewriting."""

from __future__ import annotations

import re
from typing import Any

from declaration_qa.rag.query_rewriter.schema import DEFAULT_RETRIEVAL_TOP_K


INTENT_REWRITE_TERMS: dict[str, list[str]] = {
    "material_requirements": [
        "申请材料 材料清单 必要材料 提交材料",
        "原件 复印件 电子材料 材料要求",
    ],
    "process_steps": [
        "办理流程 办理环节 申请流程 办事指南",
        "受理 审查 决定 办结",
    ],
    "eligibility": [
        "办理条件 申请条件 受理条件",
        "申请人资格 办理对象",
    ],
    "fee_time_limit": [
        "法定办结时限 承诺办结时限 办理时限",
        "多久办结 几天拿结果 审批结果",
    ],
    "office_location": [
        "办理地点 服务窗口 地址",
        "窗口地址 办事大厅",
    ],
    "contact": [
        "咨询电话 投诉电话 监督电话 联系方式",
        "咨询方式 投诉监督",
    ],
    "online_apply": [
        "网上办理 在线办理 申报入口",
        "安徽政务服务网 线上提交",
    ],
    "policy_basis": [
        "政策依据 法律依据 法规依据",
        "设定依据 办理依据",
    ],
    "form_template": [
        "申请表 表格 模板 样表",
        "空白表格 示例样表",
    ],
    "knowledge_qa": [
        "办事指南 申请要求",
    ],
    "other": [],
}


def normalize_text(value: Any) -> str:
    """Normalize text to compact whitespace."""
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_strategy_name(value: Any) -> str:
    """Normalize an LLM strategy name."""
    text = re.sub(r"[^a-zA-Z0-9_:-]+", "_", str(value or "llm_rewrite").strip().lower())
    return text[:40] or "llm_rewrite"


def normalize_top_k(value: Any) -> int:
    """Clamp per-query top_k."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_RETRIEVAL_TOP_K
    return max(1, min(6, number))


def join_query_parts(*parts: Any) -> str:
    """Join non-empty query fragments."""
    return normalize_text(" ".join(str(part) for part in parts if str(part or "").strip()))


def selected_ticket_scope_text(state: dict[str, Any]) -> str:
    """Extract item scope from current ticket context."""
    apply_context = state.get("apply_context") or {}
    if not isinstance(apply_context, dict) or not apply_context:
        return ""
    payload = apply_context.get("payload") if isinstance(apply_context.get("payload"), dict) else {}
    parts = [
        ticket_title(apply_context, state),
        payload.get("official_item_name"),
        apply_context.get("item_name"),
        payload.get("item_name"),
        apply_context.get("title"),
        payload.get("title"),
        payload.get("catalog_name"),
        apply_context.get("source_file"),
        payload.get("source_file"),
        apply_context.get("item_code"),
        payload.get("item_code"),
        apply_context.get("biz_type"),
    ]
    return normalize_text(" ".join(unique_strings([str(item) for item in parts if item])))


def has_item_scope_filter(filters: dict[str, Any]) -> bool:
    """Return whether metadata filters already include item scope."""
    return any(filters.get(key) for key in ("item_code", "ssqd_code", "ssqd_id", "item_name"))


def selected_ticket_fact_text(state: dict[str, Any]) -> str:
    """Extract allowed ticket facts for retrieval."""
    ticket_fact_type = str(state.get("ticket_fact_type") or "none")
    if ticket_fact_type == "status":
        business = state.get("business_status") or state.get("business_snapshot") or {}
        parts = [business.get("node_name"), business.get("current_node"), business.get("status")]
    elif ticket_fact_type in {"rejection", "correction"}:
        business = state.get("reject_info") or state.get("business_snapshot") or {}
        parts = [
            business.get("reject_reason"),
            business.get("material_name"),
            business.get("reject_code"),
            business.get("material_code"),
            business.get("audit_node"),
        ]
    else:
        parts = []
    return normalize_text(" ".join(str(item) for item in parts if item))


def ticket_title(*sources: dict[str, Any]) -> str:
    """Extract current item title."""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("title", "item_name", "catalog_name"):
            title = str(source.get(key) or "").strip()
            if title:
                return title
        payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
        for key in ("official_item_name", "item_name", "title", "catalog_name"):
            title = str(payload.get(key) or "").strip()
            if title:
                return title
    for source in sources:
        if isinstance(source, dict):
            title = str(source.get("biz_type") or "").strip()
            if title:
                return title
    return ""


def unique_strings(values: list[str]) -> list[str]:
    """Deduplicate strings in first-seen order."""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
