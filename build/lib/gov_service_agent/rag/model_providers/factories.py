"""按配置创建模型提供方。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.core.config import GovServiceAgentSettings
from gov_service_agent.rag.model_providers.answers import OpenAIAnswerGenerator
from gov_service_agent.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from gov_service_agent.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider
from gov_service_agent.rag.model_providers.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from gov_service_agent.rag.model_providers.embeddings.qwen3_vl_embedding_provider import Qwen3VLEmbeddingProvider
from gov_service_agent.rag.model_providers.rerankers.dashscope_reranker import DashScopeReranker
from gov_service_agent.rag.model_providers.rerankers.heuristic_reranker import HeuristicReranker
from gov_service_agent.rag.model_providers.rerankers.reranker import Reranker
from gov_service_agent.rag.prompting import DeterministicAnswerGenerator
from gov_service_agent.rag.question_resolver.llm import LLMQuestionResolver
from gov_service_agent.rag.question_resolver.rule import RuleQuestionResolver
from gov_service_agent.rag.query_rewriter.llm import LLMQueryRewriter
from gov_service_agent.rag.query_rewriter.rule import RuleQueryRewriter


def build_embedding_provider(settings: GovServiceAgentSettings) -> EmbeddingProvider:
    """按配置创建向量化提供方。"""
    provider = settings.embedding_provider.strip().lower()
    if provider in {"qwen3-vl", "qwen3_vl", "qwen3-vl-embedding", "dashscope-qwen3-vl"}:
        return Qwen3VLEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model or "qwen3-vl-embedding",
            dimension=settings.embedding_dimensions,
            image_res_level=settings.embedding_image_res_level,
        )
    if provider in {"openai", "dashscope"}:
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_api_base,
            dimension=settings.embedding_dimensions if settings.embedding_dimensions > 0 else None,
            timeout_seconds=settings.external_request_timeout_seconds,
        )
    if provider == "hashing":
        return HashEmbeddingProvider(dimension=settings.embedding_dimensions or 96)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def build_reranker(settings: GovServiceAgentSettings) -> Reranker:
    """按配置创建重排器。"""
    provider = settings.rerank_provider.strip().lower()
    if provider in {"dashscope", "sdk"}:
        return DashScopeReranker(
            api_key=settings.rerank_api_key,
            model=settings.rerank_model,
        )
    if provider == "heuristic":
        return HeuristicReranker()
    raise ValueError(f"Unsupported rerank provider: {settings.rerank_provider}")


def build_answer_generator(settings: GovServiceAgentSettings):
    """按配置创建回答生成器。"""
    provider = settings.llm_provider.strip().lower()
    if provider in {"openai", "dashscope"}:
        return OpenAIAnswerGenerator(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.openai_api_base,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.external_request_timeout_seconds,
        )
    if provider == "deterministic":
        return DeterministicAnswerGenerator()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def build_question_resolver(settings: GovServiceAgentSettings):
    """按配置创建问题解析器。"""
    if settings.llm_provider.strip().lower() in {"openai", "dashscope"} and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "model": settings.question_resolver_model or settings.llm_model,
            "temperature": 0.0,
            "timeout": settings.external_request_timeout_seconds,
            "max_retries": 1,
        }
        if settings.openai_api_base:
            kwargs["base_url"] = settings.openai_api_base
        return LLMQuestionResolver(ChatOpenAI(**kwargs), fallback_resolver=RuleQuestionResolver())
    return RuleQuestionResolver()


def build_query_rewriter(settings: GovServiceAgentSettings):
    """按配置创建多路 Query Rewrite。"""
    if settings.llm_provider.strip().lower() in {"openai", "dashscope"} and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "model": settings.query_rewriter_model or settings.llm_model,
            "temperature": 0.0,
            "timeout": settings.external_request_timeout_seconds,
            "max_retries": 1,
        }
        if settings.openai_api_base:
            kwargs["base_url"] = settings.openai_api_base
        return LLMQueryRewriter(ChatOpenAI(**kwargs), fallback_rewriter=RuleQueryRewriter())
    return RuleQueryRewriter()
