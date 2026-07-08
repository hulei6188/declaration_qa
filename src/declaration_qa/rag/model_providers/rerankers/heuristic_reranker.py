"""启发式重排器。"""

from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import RetrievedSource


class HeuristicReranker:
    """基于词面重叠和检索分数的启发式重排器。"""

    name = "heuristic"

    def rerank(
        self,
        query: str,
        sources: list[Any],
        filters: dict[str, Any],
        *,
        top_n: int,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> list[Any]:
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
