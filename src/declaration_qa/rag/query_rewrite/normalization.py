"""两阶段 Query Rewrite 输出归一化。"""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.query_rewrite.helpers import (
    join_query_parts,
    normalize_strategy_name,
    normalize_text,
    normalize_top_k,
    question_item,
)
from declaration_qa.rag.query_rewrite.schema import (
    MAX_RETRIEVAL_QUERY_COUNT,
    QueryRewrite,
    QueryRewriteStage1,
    RetrievalQuery,
)
from declaration_qa.rag.retrieval_plan import infer_retrieval_plan


def normalize_stage1_rewrite(payload: dict[str, Any], state: dict[str, Any], *, provider: str) -> QueryRewriteStage1:
    """归一化第一阶段问题消解和拆分结果。"""
    raw_questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    questions = []
    seen = set()
    for item in raw_questions:
        if not isinstance(item, dict):
            item = {"query": str(item or "")}
        raw_original = item.get("original_query") or item.get("query") or item.get("resolved_query") or ""
        original_query = normalize_question_text(raw_original)
        if not original_query or original_query in seen:
            continue
        seen.add(original_query)
        retrieval_query = normalize_question_text(
            item.get("retrieval_query") or item.get("resolved_query") or item.get("query") or original_query
        )
        raw_plan = item.get("retrieval_plan") if isinstance(item.get("retrieval_plan"), dict) else {}
        retrieval_plan = infer_retrieval_plan(
            metadata_needs=raw_plan.get("metadata_needs"),
        )
        raw_retrieval_queries = item.get("retrieval_queries") or item.get("queries") or []
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "original_query": original_query,
                "query": original_query,
                "retrieval_query": retrieval_query or original_query,
                "retrieval_plan": retrieval_plan,
                "retrieval_queries": normalize_retrieval_query_payloads(raw_retrieval_queries),
                "query_rewrite_provider": provider,
                "query_rewrite_reason": normalize_text(payload.get("reason") or "")[:160],
                "is_expanded": bool(retrieval_query and retrieval_query != original_query),
                "expansion_reason": normalize_text(item.get("expansion_reason") or payload.get("reason") or "")[:160],
            }
        )
    fallback_query = str(state.get("normalized_query") or state.get("query") or "").strip()
    if not questions and fallback_query:
        fallback = question_item(fallback_query, state)
        fallback["query_rewrite_provider"] = provider
        fallback["query_rewrite_reason"] = "fallback_single_question"
        questions = [fallback]
    return QueryRewriteStage1(
        is_multi_question=len(questions) > 1,
        questions=questions,
        reason=normalize_text(payload.get("reason") or "")[:160],
        provider=provider,
    )


def normalize_query_rewrite(
    payload: dict[str, Any],
    state: dict[str, Any],
    *,
    provider: str,
    reason: str = "",
) -> QueryRewrite:
    """归一化第二阶段检索 query 改写结果。"""
    raw_queries = payload.get("queries") if isinstance(payload.get("queries"), list) else []
    return query_rewrite_from_payloads(
        raw_queries,
        state,
        provider=provider,
        reason=reason or str(payload.get("reason") or "").strip(),
    )


def query_rewrite_from_payloads(
    raw_queries: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    provider: str,
    reason: str,
) -> QueryRewrite:
    """从第一阶段 query 草稿生成最终检索 query。"""
    base_query = normalize_text(str(state.get("retrieval_query") or state.get("normalized_query") or state.get("query") or ""))
    scope_text = ""
    fact_text = ""
    queries: list[RetrievalQuery] = []
    seen = set()
    for item in raw_queries:
        if not isinstance(item, dict):
            item = {"text": str(item or "")}
        text = normalize_text(item.get("text") or item.get("query") or "")
        text = enrich_query_text(text, scope_text=scope_text, fact_text=fact_text)
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(
            RetrievalQuery(
                id=f"rq{len(queries) + 1}",
                text=text,
                strategy=normalize_strategy_name(item.get("strategy") or "stage2_rewrite"),
                top_k=normalize_top_k(item.get("top_k")),
            )
        )
        if len(queries) >= MAX_RETRIEVAL_QUERY_COUNT:
            break
    return QueryRewrite(
        base_query=base_query,
        scope_text=scope_text,
        fact_text=fact_text,
        queries=queries,
        provider=provider,
        reason=normalize_text(reason)[:160],
    )


def normalize_retrieval_query_payloads(raw_queries: Any) -> list[dict[str, Any]]:
    """归一化第一阶段产生的 query 草稿。"""
    if not isinstance(raw_queries, list):
        return []
    output: list[dict[str, Any]] = []
    seen = set()
    for item in raw_queries:
        if not isinstance(item, dict):
            item = {"text": str(item or "")}
        text = normalize_text(item.get("text") or item.get("query") or "")
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(
            {
                "id": f"rq{len(output) + 1}",
                "text": text,
                "strategy": normalize_strategy_name(item.get("strategy") or "stage2_rewrite"),
                "top_k": normalize_top_k(item.get("top_k")),
            }
        )
        if len(output) >= MAX_RETRIEVAL_QUERY_COUNT:
            break
    return output


def enrich_query_text(text: str, *, scope_text: str, fact_text: str) -> str:
    """把第二阶段才可确定的范围和办件事实补入 query。"""
    parts = []
    if scope_text and scope_text not in text:
        parts.append(scope_text)
    if fact_text and fact_text not in text:
        parts.append(fact_text)
    parts.append(text)
    return join_query_parts(*parts)


def normalize_question_text(value: Any) -> str:
    """归一化子问题文本。"""
    return normalize_text(value).strip(" \t\r\n。；;？?")
