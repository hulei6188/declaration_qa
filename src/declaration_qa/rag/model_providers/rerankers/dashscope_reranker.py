"""DashScope SDK reranker."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from declaration_qa.core.domain import RetrievedSource


class DashScopeReranker:
    """DashScope SDK reranker adapter."""

    name = "dashscope"
    default_instruct = "Given a web search query, retrieve relevant passages that answer the query."

    def __init__(self, *, api_key: str = "", model: str = "", instruct: str = "") -> None:
        """初始化DashScopeReranker。"""
        self.api_key = api_key
        self.model = model or "qwen3-rerank"
        self.instruct = instruct or self.default_instruct

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[Any]:
        """按查询相关性重新排序候选来源。"""
        if not sources:
            return []
        dashscope = import_dashscope()
        if self.api_key:
            dashscope.api_key = self.api_key
        response = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=[source.chunk.content for source in sources],
            top_n=max(1, int(top_n)),
            instruct=self.instruct,
        )
        if response_value(response, "status_code") != HTTPStatus.OK:
            code = response_value(response, "code", "")
            message = response_value(response, "message", "")
            raise RuntimeError(f"DashScope rerank failed: {code} {message}".strip())
        output = response_value(response, "output", {}) or {}
        results = response_value(output, "results", []) or response_value(response, "results", [])
        reranked = []
        for item in results or []:
            index = int(response_value(item, "index", -1))
            score = float(response_value(item, "score", None) or response_value(item, "relevance_score", 0.0) or 0.0)
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


def response_value(payload: Any, key: str, default: Any = None) -> Any:
    """Read a value from DashScope SDK response objects or dictionaries."""
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def import_dashscope():
    """延迟导入 DashScope SDK。"""
    import dashscope

    return dashscope
