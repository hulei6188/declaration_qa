"""检索层多路 Query Rewrite。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from declaration_qa.rag_engine import normalize_text


DEFAULT_RETRIEVAL_TOP_K = 3

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


@dataclass(frozen=True)
class RetrievalQuery:
    """单路检索 query。"""

    id: str
    text: str
    strategy: str
    top_k: int = DEFAULT_RETRIEVAL_TOP_K

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "strategy": self.strategy,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class QueryRewrite:
    """多路检索 query 改写结果。"""

    base_query: str
    scope_text: str
    fact_text: str
    requires_ticket_context: bool
    ticket_fact_type: str
    retrieval_focus: str
    queries: list[RetrievalQuery]
    provider: str = "rule"

    def payload(self) -> dict[str, Any]:
        return {
            "base_query": self.base_query,
            "scope_text": self.scope_text,
            "fact_text": self.fact_text,
            "requires_ticket_context": self.requires_ticket_context,
            "ticket_fact_type": self.ticket_fact_type,
            "retrieval_focus": self.retrieval_focus,
            "queries": [query.payload() for query in self.queries],
            "provider": self.provider,
        }


class RuleQueryRewriter:
    """规则型多路 Query Rewrite，用于稳定增强召回。"""

    name = "rule"

    def rewrite(self, state: dict[str, Any]) -> QueryRewrite:
        """基于任务语义和当前事项范围生成多路检索 query。"""
        base_query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
        retrieval_focus = str(state.get("retrieval_focus") or state.get("intent") or "other")
        ticket_fact_type = str(state.get("ticket_fact_type") or "none")
        requires_ticket_context = bool(state.get("requires_ticket_context"))
        scope_text = "" if has_item_scope_filter(state.get("retrieval_filter") or {}) else selected_ticket_scope_text(state)
        fact_text = selected_ticket_fact_text(state) if requires_ticket_context else ""

        candidates = [
            ("scoped_original", join_query_parts(scope_text, fact_text, base_query)),
        ]
        for index, terms in enumerate(INTENT_REWRITE_TERMS.get(retrieval_focus, INTENT_REWRITE_TERMS["other"]), start=1):
            candidates.append((f"intent_terms_{index}", join_query_parts(scope_text, fact_text, terms, base_query)))

        queries: list[RetrievalQuery] = []
        seen = set()
        for strategy, text in candidates:
            normalized = normalize_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            queries.append(
                RetrievalQuery(
                    id=f"rq{len(queries) + 1}",
                    text=normalized,
                    strategy=strategy,
                    top_k=DEFAULT_RETRIEVAL_TOP_K,
                )
            )
            if len(queries) >= 3:
                break

        return QueryRewrite(
            base_query=base_query,
            scope_text=scope_text,
            fact_text=fact_text,
            requires_ticket_context=requires_ticket_context,
            ticket_fact_type=ticket_fact_type,
            retrieval_focus=retrieval_focus,
            queries=queries,
            provider=self.name,
        )


def join_query_parts(*parts: Any) -> str:
    """拼接 query 片段并清理空值。"""
    return normalize_text(" ".join(str(part) for part in parts if str(part or "").strip()))


def selected_ticket_scope_text(state: dict[str, Any]) -> str:
    """从当前办件提取事项范围，不包含节点、退回原因等工单事实。"""
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
    """判断检索过滤是否已包含事项级范围。"""
    return any(filters.get(key) for key in ("item_code", "ssqd_code", "ssqd_id", "item_name"))


def selected_ticket_fact_text(state: dict[str, Any]) -> str:
    """提取允许用于检索的工单事实，仅在需要工单上下文时使用。"""
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
    """提取当前事项标题。"""
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("title", "item_name", "itemName", "catalog_name", "catalogName"):
            title = str(source.get(key) or "").strip()
            if title:
                return title
        payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
        for key in ("official_item_name", "item_name", "title", "itemName", "catalog_name", "catalogName"):
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
    """按出现顺序去重字符串。"""
    seen = set()
    output = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
