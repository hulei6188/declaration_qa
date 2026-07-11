import json
from pathlib import Path

from declaration_qa.application.agent import rule_knowledge_queries
from declaration_qa.application.tools.knowledge import KnowledgeSearchInput
from declaration_qa.rag.router import classify_knowledge_intent


DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "materials"
    / "curated"
    / "declaration_qa"
    / "2026-06-15-ahzwfw-real-40-topics"
)
DATASET_PATH = DATASET_DIR / "rag_evaluation_dataset.json"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
MIN_INTENT_ACCURACY = 0.95


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_router_accuracy_dataset_covers_real_ahzwfw_40_topic_documents():
    manifest = load_json(MANIFEST_PATH)
    dataset = load_json(DATASET_PATH)
    manifest_docs = {
        Path(document["files"]["markdown"]).name
        for document in manifest["documents"]
        if document.get("files", {}).get("markdown")
    }
    case_docs = {
        tag.removeprefix("doc:")
        for case in dataset["cases"]
        for tag in case.get("tags") or []
        if tag.startswith("doc:")
    }

    assert len(manifest_docs) == 40
    assert len(dataset["cases"]) == 200
    assert case_docs == manifest_docs


def test_public_knowledge_intent_accuracy_on_real_ahzwfw_dataset():
    cases = load_json(DATASET_PATH)["cases"]
    matches = [
        classify_knowledge_intent(str(case.get("query") or "")) == str(case.get("expected_intent") or "")
        for case in cases
    ]

    assert sum(matches) / len(matches) >= MIN_INTENT_ACCURACY


def test_rule_agent_queries_never_create_ticket_binding_fields():
    for case in load_json(DATASET_PATH)["cases"]:
        query = str(case.get("query") or "")
        queries = rule_knowledge_queries({"query": query, "normalized_query": query})
        payload = KnowledgeSearchInput.model_validate({"queries": queries}).model_dump()
        serialized = json.dumps(payload, ensure_ascii=False)

        assert payload["queries"]
        assert "requires_ticket_context" not in serialized
        assert "apply_id" not in serialized
        assert "ticket_context" not in serialized
