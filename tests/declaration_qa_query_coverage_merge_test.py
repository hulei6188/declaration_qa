from __future__ import annotations

from declaration_qa.application.helpers.retrieval_sources import (
    merge_accumulated_source_payloads,
    merge_query_results_with_coverage,
)
from declaration_qa.application.workflow import DeclarationQAWorkflow
from declaration_qa.core.domain import KnowledgeChunk, RetrievalResult, RetrievedSource
from declaration_qa.rag.retrieval_plan import normalize_retrieval_plan


def test_query_coverage_merge_keeps_each_query_and_deduplicates_shared_chunk():
    shared = source("shared", "两个问题都相关", 0.99)
    material = source("material", "申请材料清单", 0.80)
    time_limit = source("time", "办理时限为5个工作日", 0.70)
    queries = [
        {"id": "q1", "text": "申请材料", "strategy": "hybrid"},
        {"id": "q2", "text": "办理时限", "strategy": "hybrid"},
    ]
    results = [
        RetrievalResult([shared, material], {}, {}),
        RetrievalResult([shared, time_limit], {}, {}),
    ]

    merged, status = merge_query_results_with_coverage(
        queries,
        results,
        search_round=1,
        min_per_query=1,
        max_sources=3,
    )

    assert [item.chunk.id for item in merged] == ["shared", "material", "time"]
    shared_metadata = merged[0].chunk.metadata
    assert shared_metadata["matched_query_ids"] == ["r1:q1", "r1:q2"]
    assert shared_metadata["best_local_rank"] == 1
    assert status["mode"] == "per_query_rerank_coverage_merge"
    assert all(item["minimum_met"] for item in status["queries"])


def test_query_coverage_merge_uses_local_rank_round_robin_not_cross_query_score():
    queries = [
        {"id": "materials", "text": "申请材料", "strategy": "hybrid"},
        {"id": "time", "text": "办理时限", "strategy": "hybrid"},
    ]
    results = [
        RetrievalResult(
            [
                source("material-1", "材料一", 0.99),
                source("material-2", "材料二", 0.98),
                source("material-3", "材料三", 0.97),
            ],
            {},
            {},
        ),
        RetrievalResult([source("time-1", "办理时限", 0.61)], {}, {}),
    ]

    merged, _ = merge_query_results_with_coverage(
        queries,
        results,
        search_round=1,
        min_per_query=1,
        max_sources=3,
    )

    assert [item.chunk.id for item in merged] == ["material-1", "time-1", "material-2"]


def test_accumulated_merge_preserves_prior_round_coverage_without_global_rerank():
    round_one = payload(source("materials", "完整材料清单", 0.72), round_no=1, query_id="materials")
    round_two = payload(source("time", "办理时限", 0.95), round_no=2, query_id="time")
    round_two_similar = payload(source("time-more", "承诺办理时限", 0.94), round_no=2, query_id="time")

    merged = merge_accumulated_source_payloads(
        [round_one, round_two, round_two_similar],
        min_per_query=1,
        max_sources=2,
    )

    assert [item["chunk_id"] for item in merged] == ["materials", "time"]
    assert merged[0]["metadata"]["matched_query_ids"] == ["r1:materials"]
    assert merged[1]["metadata"]["matched_query_ids"] == ["r2:time"]


def test_accumulated_merge_prioritizes_recent_rounds_when_limit_cannot_cover_all_queries():
    round_one = payload(source("round-1", "第一轮证据", 0.99), round_no=1, query_id="q1")
    round_two = payload(source("round-2", "第二轮证据", 0.80), round_no=2, query_id="q2")
    round_three = payload(source("round-3", "第三轮证据", 0.70), round_no=3, query_id="q3")

    merged = merge_accumulated_source_payloads(
        [round_one, round_two, round_three],
        min_per_query=1,
        max_sources=2,
    )

    assert [item["chunk_id"] for item in merged] == ["round-3", "round-2"]


def test_accumulated_merge_prioritizes_recent_rounds_when_multi_item_quotas_exceed_limit():
    sources = [
        payload(source(f"round-{round_no}-{index}", f"第{round_no}轮证据{index}", 0.9), round_no=round_no, query_id=f"q{round_no}")
        for round_no in (1, 2)
        for index in (1, 2)
    ]

    merged = merge_accumulated_source_payloads(sources, min_per_query=2, max_sources=3)

    assert [item["chunk_id"] for item in merged] == ["round-2-1", "round-2-2", "round-1-1"]


def test_retrieval_plan_preserves_only_supported_query_strategy():
    assert normalize_retrieval_plan({"retrieval_strategy": "keyword"})["retrieval_strategy"] == "keyword"
    assert normalize_retrieval_plan({"retrieval_strategy": "semantic"})["retrieval_strategy"] == "semantic"
    assert normalize_retrieval_plan({"retrieval_strategy": "invented"})["retrieval_strategy"] == "hybrid"


def test_workflow_uses_per_query_results_instead_of_global_rewrite_rerank():
    class PerQueryRag:
        rerank_score_threshold = 0.7
        rerank_max_sources = 20
        rerank_top_n_per_query = 2

        def __init__(self):
            self.queries = []

        def retrieve_many(self, queries, filters, *, retrieval_plan):
            self.queries = list(queries)
            return [
                RetrievalResult(
                    [
                        source("material-1", "材料一", 0.95),
                        source("material-2", "材料二", 0.90),
                    ],
                    dict(filters),
                    {"batch_mode": True, "channels": {"rerank": 2}},
                ),
                RetrievalResult(
                    [source("time-1", "办理时限", 0.71)],
                    dict(filters),
                    {"batch_mode": True, "channels": {"rerank": 1}},
                ),
            ]

        def retrieve_rewrites(self, *args, **kwargs):
            raise AssertionError("global rewrite rerank must not be used")

    rag = PerQueryRag()
    workflow = DeclarationQAWorkflow(
        rag_engine=rag,
        min_sources_per_query=1,
        max_accumulated_sources=2,
    )

    result = workflow.multi_retrieve(
        {
            "query": "申请材料和办理时限",
            "normalized_query": "申请材料和办理时限",
            "retrieval_queries": [
                {"id": "materials", "text": "申请材料", "strategy": "hybrid"},
                {"id": "time", "text": "办理时限", "strategy": "hybrid"},
            ],
            "knowledge_search_round": 1,
        },
        {},
    )

    assert [item["top_k"] for item in rag.queries] == [2, 2]
    assert [item.chunk.id for item in result.sources] == ["material-1", "time-1"]
    assert result.status["retrieval_mode"] == "per_query_batch_rerank"
    assert result.status["coverage_merge"]["selected_count"] == 2


def source(chunk_id: str, content: str, rerank_score: float) -> RetrievedSource:
    return RetrievedSource(
        chunk=KnowledgeChunk(
            id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            parent_id="",
            chunk_type="child",
            title=content,
            content=content,
        ),
        score=rerank_score,
        rerank_score=rerank_score,
    )


def payload(item: RetrievedSource, *, round_no: int, query_id: str) -> dict:
    value = item.payload()
    value["metadata"] = {
        "retrieval_matches": [
            {
                "match_id": f"r{round_no}:{query_id}",
                "search_round": round_no,
                "query_id": query_id,
                "query": query_id,
                "strategy": "hybrid",
                "local_rank": 1,
            }
        ]
    }
    return value
