"""检索层多路 Query Rewrite。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from declaration_qa.rag.router import parse_json_object


DEFAULT_RETRIEVAL_TOP_K = 3
MAX_RETRIEVAL_QUERY_COUNT = 3


QUERY_REWRITER_PROMPT = """你是政务问答系统的【检索 Query 改写器】，只能返回 JSON 对象，不回答业务问题，不生成最终答案。

【你的任务】
基于用户子问题、检索焦点、可用工单事实和事项范围，生成 1 到 3 条适合 RAG 检索的中文 query。

【严格边界】
1. 你只生成 retrieval queries，不生成 metadata filter；metadata filter 已由 Python 根据工单元数据严格生成。
2. 如果 context.metadata_filter_has_item_scope 为 true，说明系统已经用 item_code/item_name/ssqd_code/ssqd_id 锁定事项范围。此时不要把 scope_text 里的事项名称、事项编码重复塞进 query。
3. 如果 context.metadata_filter_has_item_scope 为 false 且 scope_text 非空，每条 query 都应自然包含 scope_text，用来避免跨事项召回。
4. 只有 requires_ticket_context 为 true 时，才允许使用 fact_text；fact_text 为空时禁止编造工单状态、退回原因、材料名称。
5. 不要输出身份证号、手机号、姓名等个人敏感信息；不要输出业务答案。
6. query 应面向检索，可以补充同义词和官方指南表达，如“申请材料/材料清单”“法定办结时限/承诺办结时限”“办理流程/办理环节”。

【返回 JSON 格式】
{
  "queries": [
    {"text": "检索 query 1", "strategy": "semantic_rewrite", "top_k": 3},
    {"text": "检索 query 2", "strategy": "focus_terms", "top_k": 3}
  ],
  "reason": "short_english_reason"
}
"""

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
    reason: str = ""

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
            "reason": self.reason,
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
            if len(queries) >= MAX_RETRIEVAL_QUERY_COUNT:
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
            reason="rule_rewrite",
        )


class LLMQueryRewriter:
    """基于大模型的多路 Query Rewrite，失败时使用规则兜底。"""

    name = "llm"

    def __init__(self, llm: Any, *, fallback_rewriter: RuleQueryRewriter | None = None) -> None:
        self.llm = llm
        self.fallback_rewriter = fallback_rewriter or RuleQueryRewriter()

    def rewrite(self, state: dict[str, Any]) -> QueryRewrite:
        """调用大模型生成检索 query。"""
        try:
            response = self.llm.invoke(build_query_rewriter_prompt(state))
            payload = parse_json_object(str(getattr(response, "content", response)))
            rewrite = normalize_query_rewrite(payload, state, provider=self.name)
            if not rewrite.queries:
                raise ValueError("empty_query_rewrite")
            return rewrite
        except Exception as exc:
            fallback = self.fallback_rewriter.rewrite(state)
            return QueryRewrite(
                base_query=fallback.base_query,
                scope_text=fallback.scope_text,
                fact_text=fallback.fact_text,
                requires_ticket_context=fallback.requires_ticket_context,
                ticket_fact_type=fallback.ticket_fact_type,
                retrieval_focus=fallback.retrieval_focus,
                queries=fallback.queries,
                provider="rule_fallback",
                reason=f"llm_query_rewriter_fallback:{type(exc).__name__}:{fallback.reason}",
            )


def build_query_rewriter_prompt(state: dict[str, Any]) -> str:
    """构造 Query Rewrite 提示词。"""
    base_query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
    filters = state.get("retrieval_filter") or {}
    scope_text = "" if has_item_scope_filter(filters) else selected_ticket_scope_text(state)
    fact_text = selected_ticket_fact_text(state) if bool(state.get("requires_ticket_context")) else ""
    context = {
        "base_query": base_query,
        "requires_ticket_context": bool(state.get("requires_ticket_context")),
        "ticket_fact_type": str(state.get("ticket_fact_type") or "none"),
        "retrieval_focus": str(state.get("retrieval_focus") or state.get("intent") or "other"),
        "scope_text": scope_text,
        "fact_text": fact_text,
        "metadata_filter": filters,
        "metadata_filter_has_item_scope": has_item_scope_filter(filters),
    }
    return f"{QUERY_REWRITER_PROMPT}\n\n请求上下文 JSON:\n{json.dumps(context, ensure_ascii=False, sort_keys=True)}"


def normalize_query_rewrite(payload: dict[str, Any], state: dict[str, Any], *, provider: str) -> QueryRewrite:
    """标准化 LLM Query Rewrite 输出。"""
    base_query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
    retrieval_focus = str(state.get("retrieval_focus") or state.get("intent") or "other")
    ticket_fact_type = str(state.get("ticket_fact_type") or "none")
    requires_ticket_context = bool(state.get("requires_ticket_context"))
    filters = state.get("retrieval_filter") or {}
    scope_text = "" if has_item_scope_filter(filters) else selected_ticket_scope_text(state)
    fact_text = selected_ticket_fact_text(state) if requires_ticket_context else ""

    queries: list[RetrievalQuery] = []
    seen = set()
    raw_queries = payload.get("queries") if isinstance(payload.get("queries"), list) else []
    for item in raw_queries:
        if isinstance(item, dict):
            text = normalize_text(str(item.get("text") or item.get("query") or ""))
            strategy = normalize_strategy_name(item.get("strategy") or "llm_rewrite")
            top_k = normalize_top_k(item.get("top_k"))
        else:
            text = normalize_text(str(item or ""))
            strategy = "llm_rewrite"
            top_k = DEFAULT_RETRIEVAL_TOP_K
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(
            RetrievalQuery(
                id=f"rq{len(queries) + 1}",
                text=text,
                strategy=strategy,
                top_k=top_k,
            )
        )
        if len(queries) >= MAX_RETRIEVAL_QUERY_COUNT:
            break

    return QueryRewrite(
        base_query=base_query,
        scope_text=scope_text,
        fact_text=fact_text,
        requires_ticket_context=requires_ticket_context,
        ticket_fact_type=ticket_fact_type,
        retrieval_focus=retrieval_focus,
        queries=queries,
        provider=provider,
        reason=str(payload.get("reason") or "").strip()[:160],
    )


def normalize_strategy_name(value: Any) -> str:
    """标准化 LLM 输出的 query 策略名称。"""
    text = re.sub(r"[^a-zA-Z0-9_:-]+", "_", str(value or "llm_rewrite").strip().lower())
    return text[:40] or "llm_rewrite"


def normalize_top_k(value: Any) -> int:
    """限制单路检索 top_k 范围。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_RETRIEVAL_TOP_K
    return max(1, min(6, number))


def join_query_parts(*parts: Any) -> str:
    """拼接 query 片段并清理空值。"""
    return normalize_text(" ".join(str(part) for part in parts if str(part or "").strip()))


def normalize_text(value: Any) -> str:
    """将文本规范化为紧凑空白形式。"""
    return re.sub(r"\s+", " ", str(value or "").strip())


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
