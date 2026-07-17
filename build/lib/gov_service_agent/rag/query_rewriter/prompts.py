"""查询改写提示词构造。"""

from __future__ import annotations

import json
from typing import Any

from gov_service_agent.rag.query_rewriter.helpers import (
    has_item_scope_filter,
    normalize_text,
    selected_ticket_fact_text,
    selected_ticket_scope_text,
)


QUERY_REWRITER_PROMPT = """你是政务问答系统的【检索 Query 改写器】，只能返回 JSON 对象，不回答业务问题，不生成最终答案。

【你的任务】
基于用户子问题、可用工单事实和事项范围，生成 1 到 3 条适合 RAG 检索的中文 query。

【严格边界】
1. 你只生成 retrieval queries，不生成 metadata filter；metadata filter 已由 Python 根据工单元数据严格生成。
2. 如果 context.metadata_filter_has_item_scope 为 true，说明系统已经用 item_code/item_name/ssqd_code/ssqd_id 锁定事项范围。此时不要把 scope_text 里的事项名称、事项编码重复塞进 query。
3. 如果 context.metadata_filter_has_item_scope 为 false 且 scope_text 非空，每条 query 都应自然包含 scope_text，用来避免跨事项召回。
4. 只有 requires_ticket_context 为 true 时，才允许使用 fact_text；fact_text 为空时禁止编造工单状态、退回原因、材料名称。
5. 不要输出身份证号、手机号、姓名等个人敏感信息；不要输出业务答案。
6. query 应面向检索，可以补充同义词和官方指南表达，如“申请材料/材料清单”“法定办结时限/承诺办结时限”“办理流程/办理环节”。
7. 不要读取或推断结构化回填策略；完整章节回填由 Retriever 独立处理。

【返回 JSON 格式】
{
  "queries": [
    {"text": "检索 query 1", "strategy": "semantic_rewrite", "top_k": 3},
    {"text": "检索 query 2", "strategy": "focus_terms", "top_k": 3}
  ],
  "reason": "short_english_reason"
}
"""


def build_query_rewriter_prompt(state: dict[str, Any]) -> str:
    """构建查询改写提示词。"""
    base_query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
    filters = state.get("retrieval_filter") or {}
    scope_text = "" if has_item_scope_filter(filters) else selected_ticket_scope_text(state)
    fact_text = selected_ticket_fact_text(state) if bool(state.get("requires_ticket_context")) else ""
    context = {
        "base_query": base_query,
        "requires_ticket_context": bool(state.get("requires_ticket_context")),
        "ticket_fact_type": str(state.get("ticket_fact_type") or "none"),
        "scope_text": scope_text,
        "fact_text": fact_text,
        "metadata_filter": filters,
        "metadata_filter_has_item_scope": has_item_scope_filter(filters),
    }
    return f"{QUERY_REWRITER_PROMPT}\n\n请求上下文 JSON:\n{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
