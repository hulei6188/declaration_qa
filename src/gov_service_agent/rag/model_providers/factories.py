"""按配置创建模型提供方。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.core.config import GovServiceAgentSettings
from gov_service_agent.rag.model_providers.embeddings.embedding_provider import EmbeddingProvider
from gov_service_agent.rag.model_providers.embeddings.hash_embedding_provider import HashEmbeddingProvider
from gov_service_agent.rag.model_providers.embeddings.openai_embedding_provider import OpenAIEmbeddingProvider
from gov_service_agent.rag.model_providers.embeddings.qwen3_vl_embedding_provider import Qwen3VLEmbeddingProvider
from gov_service_agent.rag.model_providers.langchain_compat import create_chat_openai
from gov_service_agent.rag.model_providers.rerankers.dashscope_reranker import DashScopeReranker
from gov_service_agent.rag.model_providers.rerankers.heuristic_reranker import HeuristicReranker
from gov_service_agent.rag.model_providers.rerankers.reranker import Reranker


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
            text_model=settings.text_rerank_model,
        )
    if provider == "heuristic":
        return HeuristicReranker()
    raise ValueError(f"Unsupported rerank provider: {settings.rerank_provider}")


def build_agent_model(
    settings: GovServiceAgentSettings,
    *,
    timeout_seconds: float | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
):
    """创建支持原生 Tool Calling 的 Agent 模型；本地模式返回 None。"""
    resolved_api_key = settings.openai_api_key if api_key is None else str(api_key or "").strip()
    resolved_base_url = settings.openai_api_base if base_url is None else str(base_url or "").strip()
    if settings.llm_provider.strip().lower() not in {"openai", "dashscope", "deepseek"} or not resolved_api_key:
        return None
    resolved_model = model_name or settings.agent_model or settings.llm_model
    kwargs: dict[str, Any] = {
        "api_key": resolved_api_key,
        "model": resolved_model,
        "temperature": 0.0,
        "timeout": (
            float(timeout_seconds)
            if timeout_seconds is not None
            else settings.external_request_timeout_seconds
        ),
        "max_retries": 1,
    }
    if resolved_model.strip().lower().startswith("gpt-5.6"):
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort.strip().lower() or "none"
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return create_chat_openai(**kwargs)
