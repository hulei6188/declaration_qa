from __future__ import annotations

from gov_service_agent.api.schemas import EvaluationCaseRequest, EvaluationRunRequest
from gov_service_agent.application.evals.candidates import candidate_payload, stable_sample_value
from gov_service_agent.application.evals.evaluators import evaluate_retrieval
from gov_service_agent.application.evals.online_monitor import run_online_evaluation_cycle
from gov_service_agent.application.evals.runner import summarize_items
from gov_service_agent.application.evals.retrieval_ablation import (
    extract_agent_retrieval_queries,
    run_retrieval_ablation,
)
from gov_service_agent.core.config import GovServiceAgentSettings
from gov_service_agent.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievedSource
from gov_service_agent.core.runtime import production_readiness_errors
from gov_service_agent.rag.engine.in_memory import InMemoryRAGEngine


def source(chunk_id: str, rank_score: float) -> RetrievedSource:
    return RetrievedSource(
        chunk=KnowledgeChunk(
            id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            parent_id="",
            chunk_type="child",
            title=chunk_id,
            content=f"evidence {chunk_id}",
        ),
        score=rank_score,
        dense_score=rank_score,
        bm25_score=rank_score,
        rerank_score=rank_score,
    )


def test_fixed_recall_5_10_and_mrr_10_are_emitted_together():
    sources = [{"chunk_id": f"c-{index}", "doc_id": f"d-{index}"} for index in range(1, 11)]
    scores, _, _ = evaluate_retrieval(
        sources,
        expected_doc_ids=[],
        expected_chunk_ids=["c-3", "c-8"],
        top_k=4,
    )

    assert scores["retrieval_recall_at_5"] == 0.5
    assert scores["retrieval_recall_at_10"] == 1.0
    assert scores["retrieval_mrr_at_10"] == round(1 / 3, 6)

    summary = summarize_items(
        [
            {
                "passed": True,
                "rule_scores": scores,
                "actual_trajectory": {"calls": []},
                "latency_ms": 10,
            }
        ],
        ragas_requested=False,
    )
    assert summary["recall_at_5"] == 0.5
    assert summary["recall_at_10"] == 1.0
    assert summary["mrr_at_10"] == round(1 / 3, 6)


def test_agent_queries_are_extracted_from_real_knowledge_tool_result():
    trajectory = {
        "calls": [
            {
                "tool_name": "knowledge_search",
                "arguments": {},
                "result": {
                    "executed_queries": [
                        {"text": "二手房公积金贷款材料", "strategy": "hybrid"},
                        {"text": "二手房公积金贷款材料", "strategy": "hybrid"},
                    ]
                },
            }
        ]
    }

    assert extract_agent_retrieval_queries(trajectory) == [
        {"text": "二手房公积金贷款材料", "strategy": "hybrid", "aspect": "general"}
    ]


def test_four_retrieval_variants_use_the_same_frozen_query():
    class FakeEngine:
        def retrieve_variants(self, query, filters, *, top_k, retrieval_plan):
            assert query == "取水许可申请材料"
            return {
                "variants": {
                    "bm25": [source("wrong", 1.0), source("gold", 0.9)],
                    "vector": [source("gold", 1.0)],
                    "rrf": [source("gold", 1.0)],
                    "rrf_rerank": [source("gold", 1.0)],
                },
                "status": {"elapsed_ms": 7},
            }

    result, records = run_retrieval_ablation(
        FakeEngine(),
        queries=[{"text": "取水许可申请材料", "strategy": "hybrid"}],
        filters={"region_code": "340000"},
        expected_doc_ids=[],
        expected_chunk_ids=["gold"],
    )

    assert result["status"] == "success"
    assert set(result["variants"]) == {"bm25", "vector", "rrf", "rrf_rerank"}
    assert result["variants"]["bm25"]["metrics"]["retrieval_mrr_at_10"] == 0.5
    assert result["variants"]["rrf_rerank"]["metrics"]["retrieval_mrr_at_10"] == 1.0
    assert any(record["metric_name"] == "ablation.rrf_rerank.retrieval_recall_at_10" for record in records)


def test_in_memory_engine_exposes_real_four_stage_rankings():
    engine = InMemoryRAGEngine()
    document = KnowledgeDocument(id="doc-1", title="取水许可", content="取水许可申请材料")
    chunk = KnowledgeChunk(
        id="chunk-1",
        doc_id="doc-1",
        parent_id="",
        chunk_type="child",
        title="取水许可",
        content="取水许可申请需要提交申请书和身份证明",
        embedding_id="chunk-1",
    )
    engine.index_chunks(document, [chunk])

    result = engine.retrieve_variants("取水许可申请材料", {}, top_k=10)

    assert set(result["variants"]) == {"bm25", "vector", "rrf", "rrf_rerank"}
    assert all(result["variants"][name][0].chunk.id == "chunk-1" for name in result["variants"])


def test_online_cycle_scores_langfuse_and_backfills_sanitized_candidate(monkeypatch):
    sample = {
        "audit_id": "audit-1",
        "agent_run_uid": "run-1",
        "trace_id": "trace-1",
        "query": "手机号13800138000怎么办",
        "answer": "请提交材料",
        "sources": [{"evidence_text": "办理依据"}],
        "signals": {"fingerprint": "a" * 64},
        "trajectory": {
            "calls": [
                {
                    "tool_name": "ticket_detail_query",
                    "arguments": {"application_id": "APP-SECRET"},
                    "result": {"user_id": 99},
                }
            ]
        },
        "agent_retrieval_queries": [{"text": "办理材料", "strategy": "hybrid"}],
        "failed": False,
    }
    created = []
    scored = []

    class Store:
        def create_eval_candidate(self, payload):
            created.append(payload)
            return payload

    class Workflow:
        audit_store = Store()

    class Judge:
        def evaluate(self, **kwargs):
            return {"scores": {"faithfulness": 0.7}, "status": {"status": "success"}, "records": []}

    monkeypatch.setattr(
        "gov_service_agent.application.evals.online_monitor.sample_production_traces",
        lambda *args, **kwargs: [sample],
    )
    monkeypatch.setattr(
        "gov_service_agent.application.evals.online_monitor.score_trace",
        lambda trace_id, **payload: scored.append((trace_id, payload["name"], payload["value"])),
    )
    settings = GovServiceAgentSettings(
        GOV_SERVICE_AGENT_ONLINE_EVAL_ENABLED=True,
        GOV_SERVICE_AGENT_ONLINE_EVAL_FAITHFULNESS_THRESHOLD=0.9,
    )

    result = run_online_evaluation_cycle(settings, workflow=Workflow(), judge=Judge())

    assert result == {
        "status": "success",
        "sampled": 1,
        "scored": 1,
        "low_faithfulness": 1,
        "backfilled": 1,
        "errors": [],
    }
    assert created[0]["source"] == "online_failure"
    assert "13800138000" not in created[0]["sanitized_query"]
    sanitized_call = created[0]["trajectory_snapshot"]["calls"][0]
    assert sanitized_call["arguments"]["application_id"] != "APP-SECRET"
    assert sanitized_call["result"]["user_id"] != 99
    assert created[0]["agent_retrieval_queries"][0]["text"] == "办理材料"
    assert ("trace-1", "online.faithfulness", 0.7) in scored
    assert ("trace-1", "online.sampled", 1) in scored


def test_trace_fields_and_ablation_switch_are_part_of_the_public_schema():
    case = EvaluationCaseRequest(
        query="材料有哪些",
        source_trace_id="trace-1",
        agent_retrieval_queries=[{"text": "办理材料", "strategy": "hybrid"}],
    )
    run = EvaluationRunRequest(run_retrieval_ablation=True)

    assert case.source_trace_id == "trace-1"
    assert case.agent_retrieval_queries[0]["text"] == "办理材料"
    assert run.run_retrieval_ablation is True
    assert stable_sample_value("0" * 64) == 0.0
    assert candidate_payload(
        {
            "query": "身份证340101199001011234",
            "answer": "",
            "signals": {"fingerprint": "f" * 64},
        }
    )["sanitized_query"] == "身份证[ID_CARD]"


def test_production_online_evaluation_requires_langfuse_and_independent_judge():
    settings = GovServiceAgentSettings(
        _env_file=None,
        GOV_SERVICE_AGENT_ENV="production",
        GOV_SERVICE_AGENT_ONLINE_EVAL_ENABLED=True,
        GOV_SERVICE_AGENT_LANGFUSE_ENABLED=False,
        GOV_SERVICE_AGENT_EVAL_JUDGE_ENABLED=False,
    )

    errors = production_readiness_errors(settings)

    assert "Langfuse must be configured when online evaluation is enabled" in errors
    assert "GOV_SERVICE_AGENT_EVAL_JUDGE_ENABLED=true is required when online evaluation is enabled" in errors
