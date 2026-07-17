"""回答生成模型适配器。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from gov_service_agent.observability.langfuse import invoke_with_langfuse, stream_with_langfuse
from gov_service_agent.memory.context import format_long_term_context, format_short_term_context
from gov_service_agent.rag.model_providers.streaming import stream_chat_model_content
from gov_service_agent.rag.prompting import (
    ANSWER_PROMPT,
    MULTI_QUESTION_ANSWER_PROMPT,
    format_source_payload,
    format_sub_task_payload,
)


class OpenAIAnswerGenerator:
    """基于 OpenAI Chat API 的回答生成器。"""
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        temperature: float = 0.1,
        timeout_seconds: float = 10.0,
    ) -> None:
        """初始化OpenAIAnswerGenerator。"""
        if not api_key:
            raise ValueError("OpenAI answer generator requires an API key")
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "timeout": timeout_seconds,
            "max_retries": 1,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._llm = ChatOpenAI(**kwargs)

    def build_prompt(self, state: dict[str, Any]) -> str:
        """根据工作流状态构造回答提示词。"""
        if state.get("multi_question"):
            return MULTI_QUESTION_ANSWER_PROMPT.format(
                query=state.get("query") or "",
                short_term_context=format_short_term_context(state),
                long_term_context=format_long_term_context(state),
                sub_task_payload=format_sub_task_payload(state.get("sub_tasks") or []),
            )
        sources = state.get("retrieved_sources") or []
        source_text = format_source_payload(sources)
        business_payload = state.get("business_status") or state.get("reject_info") or {}
        return ANSWER_PROMPT.format(
            query=state.get("query") or "",
            intent=state.get("intent") or "",
            retrieval_chain=state.get("retrieval_chain") or "",
            short_term_context=format_short_term_context(state),
            long_term_context=format_long_term_context(state),
            business_snapshot=state.get("business_snapshot") or {},
            business_payload=business_payload,
            source_payload=source_text,
        )

    def generate(self, state: dict[str, Any]) -> tuple[str, str]:
        """生成非流式回答和提示词。"""
        prompt = self.build_prompt(state)
        response = invoke_with_langfuse(self._llm, prompt, state, run_name="answer_generation")
        content = getattr(response, "content", response)
        return str(content), prompt

    def stream_generate(self, state: dict[str, Any]) -> tuple[Iterable[str], str]:
        """生成流式回答片段和提示词。"""
        prompt = self.build_prompt(state)
        return stream_chat_model_content(stream_with_langfuse(self._llm, prompt, state, run_name="answer_generation")), prompt
