"""基于规则的两阶段 Query Rewrite。"""

from __future__ import annotations

from typing import Any

from declaration_qa.rag.query_rewrite.helpers import (
    INTENT_REWRITE_TERMS,
    QUERY_TERM_KEYWORDS,
    join_query_parts,
    normalize_text,
    question_item,
    split_questions,
)
from declaration_qa.rag.query_rewrite.normalization import query_rewrite_from_payloads
from declaration_qa.rag.query_rewrite.schema import DEFAULT_RETRIEVAL_TOP_K, MAX_RETRIEVAL_QUERY_COUNT, QueryRewrite, QueryRewriteStage1


class RuleTwoStageQueryRewriter:
    """用于本地兜底和测试的确定性两阶段改写器。"""

    name = "rule_two_stage"

    def resolve(self, state: dict[str, Any]) -> QueryRewriteStage1:
        """第一阶段：拆问题、消解检索文本，并生成 query 草稿。"""
        query = normalize_text(str(state.get("normalized_query") or state.get("query") or ""))
        parts = split_questions(query)
        questions = []
        for part in parts:
            item = question_item(part, state)
            item["id"] = f"q{len(questions) + 1}"
            item["retrieval_queries"] = rule_query_payloads(item["retrieval_query"])
            item["query_rewrite_provider"] = self.name
            item["query_rewrite_reason"] = "rule_two_stage"
            questions.append(item)
        return QueryRewriteStage1(
            is_multi_question=len(questions) > 1,
            questions=questions,
            reason="rule_two_stage_split" if len(questions) > 1 else "single_question",
            provider=self.name,
        )

    def rewrite(self, state: dict[str, Any]) -> QueryRewrite:
        """第二阶段：把 query 草稿结合范围和办件事实归一化为最终检索 query。"""
        planned = state.get("planned_retrieval_queries") or []
        if not planned:
            base_query = normalize_text(str(state.get("retrieval_query") or state.get("normalized_query") or state.get("query") or ""))
            planned = rule_query_payloads(base_query)
            reason = "rule_two_stage_fallback"
        else:
            reason = str(state.get("query_rewrite_reason") or "reuse_stage1_queries")
        return query_rewrite_from_payloads(
            planned,
            state,
            provider=str(state.get("query_rewrite_provider") or self.name),
            reason=reason,
        )


def rule_query_payloads(base_query: str) -> list[dict[str, Any]]:
    """按规则生成第二阶段检索 query 草稿。"""
    base_query = normalize_text(base_query)
    candidates = [("scoped_original", base_query)]
    term_keys = []
    for key, keywords in QUERY_TERM_KEYWORDS:
        if key not in term_keys and any(keyword in base_query for keyword in keywords):
            term_keys.append(key)
    term_index = 1
    for term_key in term_keys:
        for terms in INTENT_REWRITE_TERMS.get(term_key, INTENT_REWRITE_TERMS["other"]):
            candidates.append((f"plan_terms_{term_index}", join_query_parts(terms, base_query)))
            term_index += 1
    queries = []
    seen = set()
    for strategy, text in candidates:
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(
            {
                "id": f"rq{len(queries) + 1}",
                "text": normalized,
                "strategy": strategy,
                "top_k": DEFAULT_RETRIEVAL_TOP_K,
            }
        )
        if len(queries) >= MAX_RETRIEVAL_QUERY_COUNT:
            break
    return queries
