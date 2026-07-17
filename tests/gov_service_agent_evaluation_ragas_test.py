from types import SimpleNamespace

from gov_service_agent.application.evaluation import (
    calculate_metrics,
    evaluate_with_ragas,
    run_evaluation,
    summarize_ragas,
)
from gov_service_agent.application.evaluation_ragas import load_ragas_components
from gov_service_agent.storage.audit import InMemoryAuditStore


class FakeRagasEvaluator:
    def evaluate(self, **kwargs):
        assert kwargs["query"] == "外国人收养需要什么材料"
        assert kwargs["reference_answer"] == "应提交护照和来华收养子女通知书。"
        assert kwargs["retrieved_contexts"] == ["申请材料包括护照。"]
        return SimpleNamespace(
            scores={"faithfulness": 0.9, "context_recall": 0.75},
            status={"provider": "ragas", "status": "success"},
        )


def test_ragas_adapter_scores_one_grounded_evaluation_case():
    result = evaluate_with_ragas(
        FakeRagasEvaluator(),
        query="外国人收养需要什么材料",
        answer="需要提交护照。",
        retrieved_contexts=["申请材料包括护照。"],
        reference_answer="应提交护照和来华收养子女通知书。",
        reference_contexts=["完整材料清单"],
        refused=False,
        requested=True,
        unavailable_reason="",
    )

    assert result["scores"] == {"faithfulness": 0.9, "context_recall": 0.75}
    assert result["status"]["status"] == "success"


def test_ragas_adapter_skips_refused_answers_and_reports_unavailable_provider():
    refused = evaluate_with_ragas(
        FakeRagasEvaluator(),
        query="伪造材料",
        answer="不能协助。",
        retrieved_contexts=["政策证据"],
        reference_answer="",
        reference_contexts=[],
        refused=True,
        requested=True,
        unavailable_reason="",
    )
    unavailable = evaluate_with_ragas(
        None,
        query="办理材料",
        answer="回答",
        retrieved_contexts=["政策证据"],
        reference_answer="",
        reference_contexts=[],
        refused=False,
        requested=True,
        unavailable_reason="agent_model_unavailable",
    )

    assert refused["status"]["reason"] == "refused_answer"
    assert unavailable["status"] == {
        "provider": "ragas",
        "status": "unavailable",
        "reason": "agent_model_unavailable",
    }


def test_ragas_scores_are_averaged_without_changing_existing_metrics():
    items = [
        {
            "recall_hit": True,
            "refusal_match": True,
            "intent_match": None,
            "keyword_hit": True,
            "ragas_scores": {"faithfulness": 0.8, "answer_relevancy": 0.6},
            "ragas_status": {"status": "success"},
        },
        {
            "recall_hit": False,
            "refusal_match": True,
            "intent_match": None,
            "keyword_hit": False,
            "ragas_scores": {"faithfulness": 1.0, "answer_relevancy": 0.8},
            "ragas_status": {"status": "partial"},
        },
    ]

    metrics = calculate_metrics(items, ragas_requested=True)

    assert metrics["recall_at_k"] == 0.5
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["ragas"]["evaluated_case_count"] == 2
    assert metrics["ragas"]["scores"] == {"faithfulness": 0.9, "answer_relevancy": 0.7}


def test_ragas_disabled_summary_is_explicit():
    assert summarize_ragas([], requested=False, unavailable_reason="") == {
        "enabled": False,
        "status": "disabled",
        "evaluated_case_count": 0,
        "scores": {},
    }


def test_ragas_components_load_lazily_with_compatible_langchain_stack():
    components = load_ragas_components()

    assert components["version"] == "0.3.9"
    assert components["Faithfulness"] is not None
    assert components["ResponseRelevancy"] is not None


def test_ragas_evaluation_run_persists_contexts_scores_and_summary():
    store = InMemoryAuditStore()
    dataset = store.create_eval_dataset(
        name="Ragas test",
        description="",
        cases=[
            {
                "query": "外国人收养需要什么材料",
                "expected_doc_ids": ["doc-adoption"],
                "reference_answer": "应提交护照和来华收养子女通知书。",
                "reference_contexts": ["完整材料清单"],
            }
        ],
    )

    class FakeWorkflow:
        def invoke(self, request):
            assert request.query == "外国人收养需要什么材料"
            return SimpleNamespace(
                sources=[
                    SimpleNamespace(
                        doc_id="doc-adoption",
                        evidence_text="申请材料包括护照。",
                        snippet="申请材料",
                    )
                ],
                answer="需要提交护照。",
                refused=False,
                intent="material_requirements",
                audit_id="audit-ragas",
            )

    result = run_evaluation(
        workflow=FakeWorkflow(),
        store=store,
        dataset_id=dataset["dataset_id"],
        ragas_evaluator=FakeRagasEvaluator(),
        ragas_requested=True,
    )

    assert result["status"] == "succeeded"
    assert result["metrics"]["recall_at_k"] == 1.0
    assert result["metrics"]["ragas"]["scores"] == {"faithfulness": 0.9, "context_recall": 0.75}
    assert result["items"][0]["retrieved_contexts"] == ["申请材料包括护照。"]
    assert result["items"][0]["reference_contexts"] == ["完整材料清单"]
    assert result["items"][0]["ragas_status"]["status"] == "success"
