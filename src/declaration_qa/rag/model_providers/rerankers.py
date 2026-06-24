"""候选检索结果重排器。"""

from __future__ import annotations

from typing import Any, Protocol

import requests

from declaration_qa.core.domain import RetrievedSource


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
        from declaration_qa.rag.engine.scoring import metadata_match_boost
        from declaration_qa.rag.engine.text import tokenize

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
