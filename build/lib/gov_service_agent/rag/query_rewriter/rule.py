"""基于规则的查询改写器。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.rag.query_rewriter.helpers import (
    INTENT_REWRITE_TERMS,
    has_item_scope_filter,
    join_query_parts,
    normalize_text,
    selected_ticket_fact_text,
    selected_ticket_scope_text,
)
from gov_service_agent.rag.query_rewriter.schema import DEFAULT_RETRIEVAL_TOP_K, MAX_RETRIEVAL_QUERY_COUNT, QueryRewrite, RetrievalQuery


QUERY_TERM_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("material_requirements", ("材料", "要带", "提交什么", "需要什么", "清单", "补什么", "上传什么")),
    ("process_steps", ("流程", "步骤", "怎么办", "如何办理", "怎么申请", "办理环节")),
    ("eligibility", ("条件", "资格", "能不能办", "是否可以")),
    ("fee_time_limit", ("费用", "收费", "多少钱", "时限", "多久", "几天", "多少天", "拿结果", "办结")),
    ("office_location", ("地点", "窗口", "地址", "哪里办")),
    ("contact", ("电话", "咨询", "投诉", "联系方式")),
    ("online_apply", ("网上办", "入口", "链接", "在线")),
    ("policy_basis", ("依据", "政策", "法规", "文件")),
    ("form_template", ("表格", "模板", "样表", "申请表")),
)


class RuleQueryRewriter:
    """用于稳定检索召回的规则多查询改写。"""

    name = "rule"

    def rewrite(self, state: dict[str, Any]) -> QueryRewrite:
        """根据任务语义和事项范围生成检索查询。"""
        base_query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
        ticket_fact_type = str(state.get("ticket_fact_type") or "none")
        requires_ticket_context = bool(state.get("requires_ticket_context"))
        scope_text = "" if has_item_scope_filter(state.get("retrieval_filter") or {}) else selected_ticket_scope_text(state)
        fact_text = selected_ticket_fact_text(state) if requires_ticket_context else ""
        candidates = [("scoped_original", join_query_parts(scope_text, fact_text, base_query))]
        term_keys = []
        for key, keywords in QUERY_TERM_KEYWORDS:
            if key not in term_keys and any(keyword in base_query for keyword in keywords):
                term_keys.append(key)
        term_index = 1
        for term_key in term_keys:
            for terms in INTENT_REWRITE_TERMS.get(term_key, INTENT_REWRITE_TERMS["other"]):
                candidates.append((f"plan_terms_{term_index}", join_query_parts(scope_text, fact_text, terms, base_query)))
                term_index += 1
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
            queries=queries,
            provider=self.name,
            reason="rule_rewrite",
        )
