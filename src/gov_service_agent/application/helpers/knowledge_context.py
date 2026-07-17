"""纯知识检索的查询和过滤辅助函数。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.application.state import GovServiceAgentState
from gov_service_agent.rag.engine.text import normalize_text
from gov_service_agent.rag.retrieval import normalize_retrieval_filter


def knowledge_rag_query(state: GovServiceAgentState) -> str:
    """只使用冻结的用户知识问题，不读取业务工具结果。"""
    return normalize_text(
        str(
            state.get("frozen_knowledge_query")
            or state.get("normalized_query")
            or state.get("query")
            or ""
        )
    )


def base_policy_filter(state: GovServiceAgentState, *, allow_request_filter: bool = True) -> dict[str, Any]:
    """只接受请求级公开知识元数据过滤条件。"""
    filters = dict(state.get("metadata_filter") or {}) if allow_request_filter else {}
    return normalize_retrieval_filter(filters)
