"""按配置创建模型提供方。"""

from __future__ import annotations

from typing import Any

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.rag.model_providers.answers import OpenAIAnswerGenerator
from declaration_qa.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from declaration_qa.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider
from declaration_qa.rag.model_providers.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from declaration_qa.rag.model_providers.rerankers.dashscope_reranker import DashScopeReranker
from declaration_qa.rag.model_providers.rerankers.heuristic_reranker import HeuristicReranker
from declaration_qa.rag.model_providers.rerankers.reranker import Reranker
from declaration_qa.rag.prompting import DeterministicAnswerGenerator
from declaration_qa.rag.question_resolver.llm import LLMQuestionResolver
from declaration_qa.rag.question_resolver.rule import RuleQuestionResolver
from declaration_qa.rag.query_rewriter.llm import LLMQueryRewriter
from declaration_qa.rag.query_rewriter.rule import RuleQueryRewriter


def build_embedding_provider(settings: DeclarationQASettings) -> EmbeddingProvider:
    """按配置创建向量化提供方。"""
    provider = settings.embedding_provider.strip().lower()
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


def build_reranker(settings: DeclarationQASettings) -> Reranker:
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


def build_answer_generator(settings: DeclarationQASettings):
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


def build_question_resolver(settings: DeclarationQASettings):
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


def build_query_rewriter(settings: DeclarationQASettings):
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
