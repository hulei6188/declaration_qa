"""纯知识检索的查询和过滤辅助函数。"""

from __future__ import annotations

from typing import Any

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.rag.engine.text import normalize_text
from declaration_qa.rag.retrieval import normalize_retrieval_filter


def knowledge_rag_query(state: DeclarationQAState) -> str:
    """只使用冻结的用户知识问题，不读取业务工具结果。"""
    return normalize_text(
        str(
            state.get("agent_knowledge_query")
            or state.get("normalized_query")
            or state.get("query")
            or ""
        )
    )


def base_policy_filter(state: DeclarationQAState, *, allow_request_filter: bool = True) -> dict[str, Any]:
    """只接受请求级公开知识元数据过滤条件。"""
    filters = dict(state.get("metadata_filter") or {}) if allow_request_filter else {}
    return normalize_retrieval_filter(filters)
