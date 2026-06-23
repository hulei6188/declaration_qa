"""向量、重排、问答生成和问题规划模型适配器。"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Protocol

import requests

from declaration_qa.config import DeclarationQASettings
from declaration_qa.prompting import ANSWER_PROMPT, MULTI_QUESTION_ANSWER_PROMPT, DeterministicAnswerGenerator, format_sub_task_payload
from declaration_qa.question_planner import LLMQuestionPlanner, RuleQuestionPlanner
from declaration_qa.query_rewriter import LLMQueryRewriter, RuleQueryRewriter


class EmbeddingProvider(Protocol):
    """文本向量化提供方协议。"""
    name: str
    dimension: int

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量表示。"""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量表示。"""
        ...


class HashEmbeddingProvider:
    """基于哈希特征的本地向量化实现。"""
    name = "hashing"

    def __init__(self, *, dimension: int = 96) -> None:
        """初始化HashEmbeddingProvider。"""
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量表示。"""
        return hashing_embedding(text, dimension=self.dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量表示。"""
        return [self.embed_query(text) for text in texts]


class OpenAIEmbeddingProvider:
    """兼容 OpenAI Embeddings API 的向量化实现。"""
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        dimension: int | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        """初始化OpenAIEmbeddingProvider。"""
        if not api_key:
            raise ValueError("OpenAI embedding provider requires an API key")
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "timeout": timeout_seconds,
            "max_retries": 1,
            "retry_min_seconds": 1,
            "retry_max_seconds": 2,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if dimension:
            kwargs["dimensions"] = dimension
        if "dashscope.aliyuncs.com" in base_url:
            kwargs["check_embedding_ctx_length"] = False
        self._embeddings = OpenAIEmbeddings(**kwargs)
        self.dimension = int(dimension or 1536)

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量表示。"""
        return [float(value) for value in self._embeddings.embed_query(text)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量表示。"""
        return [[float(value) for value in item] for item in self._embeddings.embed_documents(texts)]


class Reranker(Protocol):
    """候选检索结果重排协议。"""
    name: str

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[Any]:
        """按查询相关性重新排序候选来源。"""
        ...


class HeuristicReranker:
    """基于词面重叠和检索分数的启发式重排器。"""
    name = "heuristic"

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[Any]:
        """按查询相关性重新排序候选来源。"""
        from declaration_qa.rag_engine import RetrievedSource, metadata_match_boost, tokenize

        query_tokens = set(tokenize(query))
        reranked = []
        for hit in sources:
            content_tokens = set(tokenize(hit.chunk.content))
            overlap = len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))
            metadata_boost = metadata_match_boost(hit.chunk, filters)
            rerank_score = min(1.0, 0.55 * overlap + 0.35 * hit.dense_score + 0.25 * hit.bm25_score + metadata_boost)
            reranked.append(
                RetrievedSource(
                    chunk=hit.chunk,
                    score=max(hit.score, rerank_score),
                    dense_score=hit.dense_score,
                    bm25_score=hit.bm25_score,
                    rerank_score=rerank_score,
                    retrieval_channel=f"{hit.retrieval_channel}+rerank",
                )
            )
        return sorted(reranked, key=lambda item: (item.rerank_score, item.score), reverse=True)[:top_n]


class HttpReranker:
    """HTTP reranker adapter for production rerank services.

    Expected response:
    {
      "results": [{"index": 0, "score": 0.92}, ...]
    }
    """

    name = "http"

    def __init__(self, *, url: str, api_key: str = "", model: str = "", timeout_seconds: int = 10) -> None:
        """初始化HttpReranker。"""
        if not url:
            raise ValueError("HTTP reranker requires DECLARATION_QA_RERANK_API_URL")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[Any]:
        """按查询相关性重新排序候选来源。"""
        from declaration_qa.rag_engine import RetrievedSource

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "query": query,
            "documents": [source.chunk.content for source in sources],
            "top_n": top_n,
        }
        response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else []
        if isinstance(data, dict) and not results and isinstance(data.get("output"), dict):
            results = data["output"].get("results") or []
        reranked = []
        for item in results or []:
            index = int(item.get("index"))
            score = float(item.get("score") or item.get("relevance_score") or 0.0)
            if index < 0 or index >= len(sources):
                continue
            source = sources[index]
            reranked.append(
                RetrievedSource(
                    chunk=source.chunk,
                    score=max(source.score, score),
                    dense_score=source.dense_score,
                    bm25_score=source.bm25_score,
                    rerank_score=score,
                    retrieval_channel=f"{source.retrieval_channel}+rerank",
                )
            )
        return sorted(reranked, key=lambda item: (item.rerank_score, item.score), reverse=True)[:top_n]


class OpenAIAnswerGenerator:
    """兼容 OpenAI Chat API 的回答生成器。"""
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
                sub_task_payload=format_sub_task_payload(state.get("sub_tasks") or []),
            )
        sources = state.get("retrieved_sources") or []
        source_text = "\n".join(f"- {item.get('title')}: {item.get('snippet')}" for item in sources)
        business_payload = state.get("business_status") or state.get("reject_info") or {}
        return ANSWER_PROMPT.format(
            query=state.get("query") or "",
            intent=state.get("intent") or "",
            retrieval_chain=state.get("retrieval_chain") or "",
            business_snapshot=state.get("business_snapshot") or {},
            business_payload=business_payload,
            source_payload=source_text,
        )

    def generate(self, state: dict[str, Any]) -> tuple[str, str]:
        """生成非流式回答和提示词。"""
        prompt = self.build_prompt(state)
        response = self._llm.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content), prompt

    def stream_generate(self, state: dict[str, Any]) -> tuple[Iterable[str], str]:
        """生成流式回答片段和提示词。"""
        prompt = self.build_prompt(state)
        return stream_chat_model_content(self._llm.stream(prompt)), prompt


def stream_chat_model_content(chunks: Iterable[Any]) -> Iterable[str]:
    """从聊天模型流式结果中抽取文本片段。"""
    for chunk in chunks:
        content = getattr(chunk, "content", chunk)
        if isinstance(content, list):
            for item in content:
                text = chat_content_part_text(item)
                if text:
                    yield text
            continue
        text = str(content or "")
        if text:
            yield text


def chat_content_part_text(value: Any) -> str:
    """将聊天内容片段转换为纯文本。"""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return str(value or "")


def build_embedding_provider(settings: DeclarationQASettings) -> EmbeddingProvider:
    """按配置创建向量化提供方。"""
    provider = settings.embedding_provider.strip().lower()
    if provider in {"openai", "compatible-openai", "dashscope"}:
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
    if provider == "http":
        return HttpReranker(
            url=settings.rerank_api_url,
            api_key=settings.rerank_api_key,
            model=settings.rerank_model,
            timeout_seconds=settings.external_request_timeout_seconds,
        )
    if provider == "heuristic":
        return HeuristicReranker()
    raise ValueError(f"Unsupported rerank provider: {settings.rerank_provider}")


def build_answer_generator(settings: DeclarationQASettings):
    """按配置创建回答生成器。"""
    provider = settings.llm_provider.strip().lower()
    if provider in {"openai", "compatible-openai", "dashscope"}:
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


def build_question_planner(settings: DeclarationQASettings):
    """按配置创建多问题拆解 planner。"""
    if settings.llm_provider.strip().lower() in {"openai", "compatible-openai", "dashscope"} and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "model": settings.question_planner_model or settings.legacy_intent_router_model or settings.llm_model,
            "temperature": 0.0,
            "timeout": settings.external_request_timeout_seconds,
            "max_retries": 1,
        }
        if settings.openai_api_base:
            kwargs["base_url"] = settings.openai_api_base
        return LLMQuestionPlanner(ChatOpenAI(**kwargs), fallback_planner=RuleQuestionPlanner())
    return RuleQuestionPlanner()


def build_query_rewriter(settings: DeclarationQASettings):
    """按配置创建多路 Query Rewrite。"""
    if settings.llm_provider.strip().lower() in {"openai", "compatible-openai", "dashscope"} and settings.openai_api_key:
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


def hashing_embedding(text: str, *, dimension: int) -> list[float]:
    """使用哈希技巧生成归一化文本向量。"""
    from declaration_qa.rag_engine import tokenize

    values = [0.0 for _ in range(dimension)]
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimension
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    """计算两个向量的余弦点积分数。"""
    left_values = list(left)
    right_values = list(right)
    return sum(a * b for a, b in zip(left_values, right_values))
