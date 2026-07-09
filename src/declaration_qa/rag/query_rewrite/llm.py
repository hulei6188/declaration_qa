"""基于 LLM 的两阶段 Query Rewrite。"""

from __future__ import annotations

from typing import Any

from declaration_qa.observability.langfuse import invoke_with_langfuse
from declaration_qa.rag.query_rewrite.normalization import normalize_question_resolution, query_rewrite_from_payloads
from declaration_qa.rag.query_rewrite.prompts import build_two_stage_query_rewrite_prompt
from declaration_qa.rag.query_rewrite.rule import RuleTwoStageQueryRewriter
from declaration_qa.rag.query_rewrite.schema import QueryRewrite, QuestionResolution
from declaration_qa.rag.router import parse_json_object


class LLMTwoStageQueryRewriter:
    """基于 LLM 的两阶段改写器，失败时使用规则兜底。"""

    name = "llm_two_stage"

    def __init__(self, llm: Any, *, fallback_rewriter: RuleTwoStageQueryRewriter | None = None) -> None:
        self.llm = llm
        self.fallback_rewriter = fallback_rewriter or RuleTwoStageQueryRewriter()

    def resolve(self, state: dict[str, Any]) -> QuestionResolution:
        """第一阶段：调用 LLM 完成问题消解、拆分和 query 草稿生成。"""
        try:
            response = invoke_with_langfuse(
                self.llm,
                build_two_stage_query_rewrite_prompt(state),
                state,
                run_name="query_rewrite.stage1",
            )
            payload = parse_json_object(str(getattr(response, "content", response)))
            resolution = normalize_question_resolution(payload, state, provider=self.name)
            if not resolution.questions:
                raise ValueError("empty_two_stage_resolution")
            return resolution
        except Exception as exc:
            fallback = self.fallback_rewriter.resolve(state)
            return QuestionResolution(
                is_multi_question=fallback.is_multi_question,
                questions=fallback.questions,
                reason=f"llm_two_stage_fallback:{type(exc).__name__}:{fallback.reason}",
                provider="rule_two_stage_fallback",
            )

    def rewrite(self, state: dict[str, Any]) -> QueryRewrite:
        """第二阶段：复用第一阶段 query 草稿，不再调用 LLM。"""
        planned = state.get("planned_retrieval_queries") or []
        if planned:
            rewrite = query_rewrite_from_payloads(
                planned,
                state,
                provider=str(state.get("query_rewrite_provider") or self.name),
                reason=str(state.get("query_rewrite_reason") or "reuse_stage1_queries"),
            )
            if rewrite.queries:
                return rewrite
        fallback = self.fallback_rewriter.rewrite(state)
        return QueryRewrite(
            base_query=fallback.base_query,
            scope_text=fallback.scope_text,
            fact_text=fallback.fact_text,
            requires_ticket_context=fallback.requires_ticket_context,
            ticket_fact_type=fallback.ticket_fact_type,
            queries=fallback.queries,
            provider="rule_two_stage_fallback",
            reason=f"missing_stage1_queries:{fallback.reason}",
        )
