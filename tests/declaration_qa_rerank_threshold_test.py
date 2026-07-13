from __future__ import annotations

from typing import Any

from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievedSource
from declaration_qa.rag.engine.in_memory import InMemoryRAGEngine


class MetadataScoreReranker:
    name = "metadata-score"

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int) -> list[RetrievedSource]:
        reranked: list[RetrievedSource] = []
        for source in sources:
            score = float((source.chunk.metadata or {}).get("test_rerank_score") or 0.0)
            reranked.append(
                RetrievedSource(
                    chunk=source.chunk,
                    score=max(float(source.score or 0.0), score),
                    dense_score=float(source.dense_score or 0.0),
                    bm25_score=float(source.bm25_score or 0.0),
                    rerank_score=score,
                    retrieval_channel=f"{source.retrieval_channel}+rerank",
                )
            )
        return sorted(reranked, key=lambda item: (item.rerank_score, item.score), reverse=True)[:top_n]


class FailingReranker:
    name = "failing"

    def rerank(self, query: str, sources: list[Any], filters: dict[str, Any], *, top_n: int):
        raise RuntimeError("request payload is too large")


def test_rerank_score_threshold_respects_per_query_top_n() -> None:
    engine = InMemoryRAGEngine(
        reranker=MetadataScoreReranker(),
        rerank_score_threshold=0.65,
        rerank_max_sources=12,
    )
    document, chunks = scored_policy_chunks([0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.50])
    engine.load_indexed_data([document], chunks)

    result = engine.retrieve("创业补贴 申请材料", {}, top_k=5)

    assert [source.chunk.id for source in result.sources] == [f"chunk-score-{index}" for index in range(5)]
    assert result.status["channels"]["rerank"] == 5
    assert result.status["channels"]["selected"] == 5
    assert result.status["score_cutoff"] == {
        "enabled": True,
        "threshold": 0.65,
        "requested_top_k": 5,
        "max_sources": 12,
        "before": 5,
        "after": 5,
        "rejected": False,
        "reason": "passed_threshold",
    }


def test_rerank_score_threshold_rejects_all_low_confidence_hits() -> None:
    engine = InMemoryRAGEngine(
        reranker=MetadataScoreReranker(),
        rerank_score_threshold=0.65,
        rerank_max_sources=12,
    )
    document, chunks = scored_policy_chunks([0.44, 0.42, 0.40])
    engine.load_indexed_data([document], chunks)

    result = engine.retrieve("创业补贴 申请材料", {}, top_k=5)

    assert result.sources == []
    assert result.status["channels"]["rerank"] == 3
    assert result.status["channels"]["selected"] == 0
    assert result.status["score_cutoff"]["rejected"] is True
    assert result.status["score_cutoff"]["after"] == 0
    assert result.status["score_cutoff"]["reason"] == "all_rerank_scores_below_threshold"


def test_rerank_failure_keeps_heuristic_sources_without_model_threshold() -> None:
    engine = InMemoryRAGEngine(
        reranker=FailingReranker(),
        rerank_score_threshold=0.7,
        rerank_max_sources=12,
    )
    document, chunks = scored_policy_chunks([0.0, 0.0, 0.0])
    engine.load_indexed_data([document], chunks)

    result = engine.retrieve("创业补贴 申请材料", {}, top_k=2)

    assert len(result.sources) == 2
    assert result.status["score_cutoff"]["enabled"] is False
    assert result.status["score_cutoff"]["reason"] == "reranker_failed_threshold_skipped"
    assert result.status["degraded"] == [
        {"stage": "rerank", "error": "RuntimeError", "message": "request payload is too large"}
    ]


def scored_policy_chunks(scores: list[float]) -> tuple[KnowledgeDocument, list[KnowledgeChunk]]:
    document = KnowledgeDocument(
        id="doc-score-threshold",
        title="创业补贴政策测试",
        content="创业补贴申请材料测试集合",
    )
    chunks = [
        KnowledgeChunk(
            id=f"chunk-score-{index}",
            doc_id=document.id,
            parent_id="",
            chunk_type="child",
            title=f"创业补贴材料 {index}",
            content=f"创业补贴 申请材料 第{index}条，需要提交政策材料说明。",
            metadata={"doc_status": "published", "test_rerank_score": score},
        )
        for index, score in enumerate(scores)
    ]
    return document, chunks
