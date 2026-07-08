"""重排器协议。"""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    """候选检索结果重排协议。"""

    name: str

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
        ...
