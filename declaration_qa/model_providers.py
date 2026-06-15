from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Protocol

import requests

from declaration_qa.config import DeclarationQASettings
from declaration_qa.prompting import ANSWER_PROMPT, DeterministicAnswerGenerator


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbeddingProvider:
    name = "hashing"

    def __init__(self, *, dimension: int = 96) -> None:
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return hashing_embedding(text, dimension=self.dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, *, api_key: str, model: str, base_url: str = "", dimension: int | None = None) -> None:
        if not api_key:
            raise ValueError("OpenAI embedding provider requires an API key")
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {"api_key": api_key, "model": model}
        if base_url:
            kwargs["base_url"] = base_url
        if dimension:
            kwargs["dimensions"] = dimension
        if "dashscope.aliyuncs.com" in base_url:
            kwargs["check_embedding_ctx_length"] = False
        self._embeddings = OpenAIEmbeddings(**kwargs)
        self.dimension = int(dimension or 1536)

    def embed_query(self, text: str) -> list[float]:
        return [float(value) for value in self._embeddings.embed_query(text)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in item] for item in self._embeddings.embed_documents(texts)]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[Any]:
        ...


class HeuristicReranker:
    name = "heuristic"

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[Any]:
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
        if not url:
            raise ValueError("HTTP reranker requires DECLARATION_QA_RERANK_API_URL")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[Any]:
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
    name = "openai"

    def __init__(self, *, api_key: str, model: str, base_url: str = "", temperature: float = 0.1) -> None:
        if not api_key:
            raise ValueError("OpenAI answer generator requires an API key")
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"api_key": api_key, "model": model, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        self._llm = ChatOpenAI(**kwargs)

    def generate(self, state: dict[str, Any]) -> tuple[str, str]:
        sources = state.get("retrieved_sources") or []
        source_text = "\n".join(f"- {item.get('title')}: {item.get('snippet')}" for item in sources[:6])
        business_payload = state.get("business_status") or state.get("reject_info") or {}
        prompt = ANSWER_PROMPT.format(
            query=state.get("query") or "",
            intent=state.get("intent") or "",
            business_payload=business_payload,
            source_payload=source_text,
        )
        response = self._llm.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content), prompt


def build_embedding_provider(settings: DeclarationQASettings) -> EmbeddingProvider:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"openai", "compatible-openai", "dashscope"}:
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_api_base,
            dimension=settings.embedding_dimensions if settings.embedding_dimensions > 0 else None,
        )
    if provider == "hashing":
        return HashEmbeddingProvider(dimension=settings.embedding_dimensions or 96)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def build_reranker(settings: DeclarationQASettings) -> Reranker:
    provider = settings.rerank_provider.strip().lower()
    if provider == "http":
        return HttpReranker(
            url=settings.rerank_api_url,
            api_key=settings.rerank_api_key,
            model=settings.rerank_model,
        )
    if provider == "heuristic":
        return HeuristicReranker()
    raise ValueError(f"Unsupported rerank provider: {settings.rerank_provider}")


def build_answer_generator(settings: DeclarationQASettings):
    provider = settings.llm_provider.strip().lower()
    if provider in {"openai", "compatible-openai", "dashscope"}:
        return OpenAIAnswerGenerator(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.openai_api_base,
            temperature=settings.llm_temperature,
        )
    if provider == "deterministic":
        return DeterministicAnswerGenerator()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def hashing_embedding(text: str, *, dimension: int) -> list[float]:
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
    left_values = list(left)
    right_values = list(right)
    return sum(a * b for a, b in zip(left_values, right_values))
