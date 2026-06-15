from fastapi.testclient import TestClient
from pathlib import Path

from declaration_qa.domain import KnowledgeDocument
from declaration_qa.rag_engine import InMemoryRAGEngine, build_seed_rag_engine
from declaration_qa.repository import PostgresDeclarationQAStore
from declaration_qa.schemas import ChatRequest
from declaration_qa.server import create_app
from declaration_qa.source_types import MATERIAL_RULE_SOURCE_TYPE
from declaration_qa.workflow import DeclarationQAWorkflow


def test_declaration_qa_entrypoint_isolated_from_existing_main():
    client = TestClient(create_app())

    response = client.get("/api/declaration-qa/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["entrypoint"] == "declaration_qa.server:app"
    assert payload["service"] == "declaration-qa"
    assert payload["audit_store"] == "InMemoryAuditStore"
    assert client.get("/api/agents").status_code == 404


def test_policy_filter_does_not_use_app_version():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="办理居住证需要哪些材料",
            user_id=1,
            region_code="340000",
            app_version="1.2.3",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.intent == "knowledge"
    assert response.refused is False
    assert "居住证" in response.answer
    assert "app_version" not in audit["retrieval_filter"]
    assert "appVersion" not in audit["retrieval_filter"]
    assert audit["retrieval_filter"]["biz_type"] == "residence_permit"
    assert response.suggested_actions[0].payload["appVersion"] == "1.2.3"


def test_status_question_requires_optional_apply_context_only_for_status_path():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(ChatRequest(query="我的健康证办理到哪一步了", user_id=1, region_code="340000"))

    assert response.intent == "status"
    assert response.refused is True
    assert response.refuse_reason == "apply_context_required_for_status"
    assert "选择对应申请单" in response.answer


def test_status_question_uses_business_status_then_rag_explanation():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="我的健康证办理到哪一步了",
            user_id=1,
            region_code="340000",
            apply_id="A1001",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.intent == "status"
    assert response.refused is False
    assert "部门审核中" in response.answer
    assert audit["business_status"]["current_node"] == "DEPT_REVIEW"
    assert audit["retrieval_filter"]["source_type"] == "process"
    assert audit["answer"] == response.answer


def test_reject_flow_maps_material_item_before_rag():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="为什么我的申请被退回了",
            user_id=1,
            region_code="340000",
            apply_id="A3003",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.intent == "reject"
    assert response.refused is False
    assert "身份证正面照片" in response.answer
    assert audit["reject_info"]["reject_code"] == "PHOTO_BLUR"
    assert audit["material_mapping"]["material_code"] == "ID_CARD_FRONT"
    assert audit["retrieval_filter"]["source_type"] == MATERIAL_RULE_SOURCE_TYPE
    assert audit["retrieval_filter"]["material_code"] == "ID_CARD_FRONT"
    assert audit["retrieval_filter"]["reject_code"] == "PHOTO_BLUR"
    assert response.sources[0].source_type == MATERIAL_RULE_SOURCE_TYPE
    assert response.sources[0].material_code == "ID_CARD_FRONT"
    assert response.sources[0].reject_code == "PHOTO_BLUR"


def test_reject_flow_falls_back_to_material_rule_without_reject_code():
    engine = InMemoryRAGEngine()
    engine.index_document(
        KnowledgeDocument(
            id="doc-id-card-material-rule",
            title="身份证正面照片材料规范",
            content="身份证正面照片不清晰时，需要重新上传边框完整、文字清楚、无遮挡、无反光的图片。",
            biz_type="id_card_replace",
            region_code="340000",
            policy_version="2025",
            source_type=MATERIAL_RULE_SOURCE_TYPE,
            material_code="ID_CARD_FRONT",
        )
    )
    workflow = DeclarationQAWorkflow(rag_engine=engine)

    response = workflow.invoke(
        ChatRequest(
            query="为什么我的申请被退回了",
            user_id=1,
            region_code="340000",
            apply_id="A3003",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.intent == "reject"
    assert response.refused is False
    assert audit["retrieval_filter"]["source_type"] == MATERIAL_RULE_SOURCE_TYPE
    assert audit["retrieval_filter"]["material_code"] == "ID_CARD_FRONT"
    assert "reject_code" not in audit["retrieval_filter"]
    assert response.sources[0].material_code == "ID_CARD_FRONT"


def test_stream_endpoint_returns_sse_events():
    client = TestClient(create_app())

    response = client.post(
        "/api/declaration-qa/chat/stream",
        json={
            "query": "办理居住证需要哪些材料",
            "user_id": 1,
            "region_code": "340000",
            "app_version": "2.0.0",
        },
    )

    assert response.status_code == 200
    assert "event: run_started" in response.text
    assert response.text.index("event: intent") < response.text.index("event: token")
    assert "event: sources" in response.text
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert response.text.index("event: audit_logged") < response.text.index("event: done")


def test_status_stream_returns_business_status_and_rag_progress():
    client = TestClient(create_app())

    response = client.post(
        "/api/declaration-qa/chat/stream",
        json={
            "query": "我的健康证办理到哪一步了",
            "user_id": 1,
            "region_code": "340000",
            "apply_id": "A1001",
        },
    )

    assert response.status_code == 200
    assert "event: business_status" in response.text
    assert "DEPT_REVIEW" in response.text
    assert "event: rag_status" in response.text
    assert "event: sources" in response.text


def test_declaration_qa_schema_file_defines_isolated_tables():
    schema_path = Path("sql/postgresql/003_declaration_qa_schema.sql")
    sql = schema_path.read_text(encoding="utf-8")

    assert "CREATE SCHEMA IF NOT EXISTS declaration_qa" in sql
    for table in [
        "declaration_qa.knowledge_bases",
        "declaration_qa.documents",
        "declaration_qa.chunks",
        "declaration_qa.index_jobs",
        "declaration_qa.sessions",
        "declaration_qa.messages",
        "declaration_qa.audit_logs",
        "declaration_qa.feedback",
        "declaration_qa.eval_datasets",
        "declaration_qa.eval_cases",
        "declaration_qa.eval_runs",
        "declaration_qa.eval_run_items",
    ]:
        assert table in sql
    assert "material_code" in sql
    assert "reject_code" in sql
    assert "audit_node" in sql
    assert "app_version" not in sql.lower()


def test_evidence_check_refuses_when_no_matching_policy_source():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="办理居住证需要哪些材料",
            user_id=1,
            region_code="110000",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.intent == "knowledge"
    assert response.refused is True
    assert response.refuse_reason == "no_matched_source"
    assert audit["retrieved_sources"] == []
    assert "人工客服" in response.answer


def test_rag_cache_reports_hit_on_repeated_query():
    engine = build_seed_rag_engine()
    filters = {
        "biz_type": "residence_permit",
        "region_code": "340000",
        "policy_version": "2025",
        "status": "published",
        "source_type": ["guide"],
    }

    first = engine.retrieve("办理居住证需要哪些材料", filters)
    second = engine.retrieve("办理居住证需要哪些材料", filters)

    assert first.status["cache"]["hit"] is False
    assert second.status["cache"]["hit"] is True
    assert second.status["cache"]["backend"] == "memory"
    assert second.status["vector_backend"] == "memory"


def test_postgres_business_adapter_reads_application_status_and_reject_tables():
    rows = [
        {
            "apply_id": "A9001",
            "user_id": 7,
            "biz_type": "id_card_replace",
            "region_code": "340000",
            "status": "rejected",
            "current_node": "PHOTO_REVIEW",
            "node_name": "照片审核",
            "reject_code": "PHOTO_BLUR",
            "reject_reason": "照片不清晰",
            "material_code": "ID_CARD_FRONT",
            "material_name": "身份证正面照片",
            "audit_node": "人工初审",
        },
        {"exists": 1},
        {
            "apply_id": "A9001",
            "biz_type": "id_card_replace",
            "region_code": "340000",
            "status": "rejected",
            "current_node": "PHOTO_REVIEW",
            "node_name": "照片审核",
        },
        {
            "apply_id": "A9001",
            "biz_type": "id_card_replace",
            "region_code": "340000",
            "reject_code": "PHOTO_BLUR",
            "reject_reason": "照片不清晰",
            "material_code": "ID_CARD_FRONT",
            "material_name": "身份证正面照片",
            "audit_node": "人工初审",
        },
    ]
    store = PostgresDeclarationQAStore.__new__(PostgresDeclarationQAStore)
    store.engine = FakeBusinessEngine(rows)

    context = store.get_apply_context(user_id=7, apply_id="A9001")
    has_access = store.verify_apply_access(user_id=7, apply_id="A9001")
    status = store.get_status(context)
    reject = store.get_reject_info(context)

    assert has_access is True
    assert context["apply_id"] == "A9001"
    assert status["current_node"] == "PHOTO_REVIEW"
    assert reject["reject_code"] == "PHOTO_BLUR"
    assert reject["material_code"] == "ID_CARD_FRONT"


class FakeBusinessEngine:
    def __init__(self, rows):
        self.rows = list(rows)

    def begin(self):
        return self

    def __enter__(self):
        return FakeBusinessConnection(self.rows)

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeBusinessConnection:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, statement, params=None):
        return FakeBusinessResult(self.rows.pop(0))


class FakeBusinessResult:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


def test_feedback_endpoint_records_user_feedback():
    client = TestClient(create_app())
    chat_response = client.post(
        "/api/declaration-qa/chat",
        json={"query": "办理居住证需要哪些材料", "user_id": 1, "region_code": "340000"},
    )
    audit_id = chat_response.json()["audit_id"]

    response = client.post(
        "/api/declaration-qa/feedback",
        json={"audit_id": audit_id, "rating": "up", "comment": "有帮助"},
    )

    assert response.status_code == 200
    assert response.json()["audit_id"] == audit_id
    assert response.json()["rating"] == "up"


def test_isolated_module_does_not_import_existing_runtime_code():
    root = Path("declaration_qa")
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))

    assert "from core" not in text
    assert "import core" not in text
    assert "from api" not in text
    assert "import api" not in text
    assert "api.main" not in text
