from fastapi.testclient import TestClient
from pathlib import Path

from declaration_qa.business import InMemoryBusinessDataAdapter
from declaration_qa.domain import KnowledgeChunk, KnowledgeDocument
from declaration_qa.rag_engine import InMemoryRAGEngine, build_seed_rag_engine
from declaration_qa.repository import PostgresDeclarationQAStore
from declaration_qa.router import LLMIntentRouter
from declaration_qa.schemas import ChatRequest
from declaration_qa.server import create_app
from declaration_qa.workflow import STREAM_TOKEN_DELAY_SECONDS, STREAM_TOKEN_SIZE, DeclarationQAWorkflow, ticket_rag_query, token_events


def test_declaration_qa_entrypoint_isolated_from_existing_main():
    client = TestClient(create_app())

    response = client.get("/api/declaration-qa/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["entrypoint"] == "declaration_qa.server:app"
    assert payload["service"] == "declaration-qa"
    assert payload["audit_store"] == "InMemoryAuditStore"
    assert client.get("/api/agents").status_code == 404


def test_ticket_list_endpoint_supports_pre_chat_selection():
    client = TestClient(create_app())

    response = client.get("/api/declaration-qa/tickets", params={"user_id": 1, "region_code": "340000"})

    assert response.status_code == 200
    assert any(item["ticket_id"] == "A1001" for item in response.json())


def test_ticket_list_matches_six_digit_region_to_twelve_digit_province_code():
    adapter = InMemoryBusinessDataAdapter()
    adapter._applications["A1001"]["region_code"] = "340000000000"

    tickets = adapter.list_applications(user_id=1, region_code="340000")

    assert any(item["ticket_id"] == "A1001" for item in tickets)


def test_knowledge_refresh_loads_persisted_chunks_without_reembedding(monkeypatch):
    from declaration_qa import server

    document = KnowledgeDocument(id="doc-refresh", title="刷新测试", content="原文")
    chunk = KnowledgeChunk(id="chunk-refresh", doc_id="doc-refresh", parent_id="", chunk_type="child", title="刷新测试", content="内容")

    class FakeStore:
        def knowledge_base_revision(self):
            return "rev-new"

        def list_indexed_knowledge(self):
            return [document], [chunk]

    class FakeRagEngine:
        def __init__(self):
            self.loaded = None

        def load_indexed_data(self, documents, chunks):
            self.loaded = (documents, chunks)

        def rebuild(self, documents):
            raise AssertionError("chat refresh must not rebuild embeddings")

    class FakeWorkflow:
        def __init__(self):
            self.audit_store = FakeStore()
            self.rag_engine = FakeRagEngine()

    fake_workflow = FakeWorkflow()
    monkeypatch.setattr(server, "workflow", fake_workflow)
    monkeypatch.setattr(server, "_knowledge_base_revision", "rev-old")

    server.ensure_knowledge_base_fresh()

    assert fake_workflow.rag_engine.loaded == ([document], [chunk])
    assert server._knowledge_base_revision == "rev-new"


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

    assert response.route == "ticket_context_unrelated"
    assert response.intent == "material_requirements"
    assert response.refused is False
    assert "居住证" in response.answer
    assert "app_version" not in audit["retrieval_filter"]
    assert "appVersion" not in audit["retrieval_filter"]
    assert "biz_type" not in audit["retrieval_filter"]
    assert response.suggested_actions[0].payload["appVersion"] == "1.2.3"
    assert response.retrieval_chain == "knowledge_retrieval"
    assert response.evidence_decision["ok"] is True


def test_stream_token_events_are_small_and_throttled():
    delays = []

    events = list(token_events("abcdefg", sleep=delays.append))

    assert [event["data"]["content"] for event in events] == ["abc", "def", "g"]
    assert STREAM_TOKEN_SIZE == 3
    assert delays == [STREAM_TOKEN_DELAY_SECONDS, STREAM_TOKEN_DELAY_SECONDS]


def test_llm_stream_generator_emits_model_tokens_and_records_answer():
    class FakeStreamingGenerator:
        def stream_generate(self, state):
            return iter(["真", "流", "式"]), "prompt-text"

    workflow = DeclarationQAWorkflow(answer_generator=FakeStreamingGenerator())
    state = workflow._initial_state(ChatRequest(query="测试", user_id=1, region_code="340000"))

    events = list(workflow.llm_generate_stream_events(state))

    assert [event["data"]["content"] for event in events if event["event"] == "token"] == ["真", "流", "式"]
    assert events[-1] == {"event": "answer_generated", "data": {"intent": "unknown", "streamed": True}}
    assert state["answer"] == "真流式"
    assert state["audit_payload"]["final_prompt"] == "prompt-text"


def test_ticket_rag_query_prefers_ticket_chinese_title():
    query = ticket_rag_query(
        {
            "apply_context": {
                "biz_type": "foreigners_group_religious_activity_temporary_site_approval",
                "payload": {"item_name": "境内外国人集体进行宗教活动临时地点审批"},
            },
            "business_snapshot": {},
            "business_status": {},
            "reject_info": {},
            "material_mapping": {},
            "biz_type": "foreigners_group_religious_activity_temporary_site_approval",
            "normalized_query": "办这个需要哪些材料？",
        }
    )

    assert query.startswith("境内外国人集体进行宗教活动临时地点审批")
    assert "办这个需要哪些材料" in query


def test_knowledge_chain_ignores_removed_metadata_filter_aliases():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="身份证照片上传规范是什么",
            user_id=1,
            region_code="340000",
            biz_type="id_card_replace",
            metadata_filter={
                "sourceType": "material_rule",
                "materialCode": "ID_CARD_FRONT",
                "rejectCode": "PHOTO_BLUR",
            },
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.route == "ticket_context_unrelated"
    assert response.intent == "knowledge_qa"
    assert response.refused is False
    assert response.retrieval_chain == "knowledge_retrieval"
    assert "biz_type" not in audit["retrieval_filter"]
    assert "source_type" not in audit["retrieval_filter"]
    assert "material_code" not in audit["retrieval_filter"]
    assert "reject_code" not in audit["retrieval_filter"]


def test_status_question_auto_selects_single_related_ticket():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(ChatRequest(query="我的健康证办理到哪一步了", user_id=1, region_code="340000"))

    assert response.route == "ticket_context_related"
    assert response.intent == "ticket_status"
    assert response.refused is False
    assert response.need_ticket_selection is False
    assert response.ticket_selection_reason == "single_candidate"
    assert response.ticket_context["ticket"]["ticket_id"] == "A1001"
    assert "部门审核中" in response.answer


def test_ticket_question_requires_selection_when_multiple_candidates():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(ChatRequest(query="我的申请办到哪了", user_id=1, region_code="340000"))

    assert response.route == "ticket_context_related"
    assert response.intent == "ticket_status"
    assert response.refused is True
    assert response.need_ticket_selection is True
    assert response.ticket_selection_reason == "multiple_candidates"
    assert len(response.tickets) >= 2
    assert "多个可能相关的申请单" in response.answer


class FakeRouterLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, prompt: str):
        self.prompt = prompt
        return type("FakeMessage", (), {"content": self.content})()


def test_llm_router_handles_ticket_question_without_keyword():
    llm = FakeRouterLLM(
        """
        {
          "route": "ticket_context_related",
          "intent": "ticket_status",
          "sub_intent": "ticket_status",
          "confidence": 0.91,
          "requires_ticket_context": true,
          "can_answer_from_knowledge": false,
          "reason": "llm_personal_ticket_status",
          "entities": {"biz_type": "health_certificate", "apply_id": "A1001"}
        }
        """
    )
    workflow = DeclarationQAWorkflow(intent_router=LLMIntentRouter(llm))

    response = workflow.invoke(
        ChatRequest(query="麻烦看一下这单现在什么情况", user_id=1, region_code="340000", apply_id="A1001")
    )

    assert response.route == "ticket_context_related"
    assert response.intent == "ticket_status"
    assert response.route_decision["entities"]["router_provider"] == "llm"
    assert "部门审核中" in response.answer


def test_llm_router_falls_back_to_rules_when_model_output_is_invalid():
    workflow = DeclarationQAWorkflow(intent_router=LLMIntentRouter(FakeRouterLLM("不是 JSON")))

    response = workflow.invoke(ChatRequest(query="办理居住证需要哪些材料", user_id=1, region_code="340000"))

    assert response.route == "ticket_context_unrelated"
    assert response.intent == "material_requirements"
    assert response.route_decision["entities"]["router_provider"] == "rule_fallback"


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

    assert response.route == "ticket_context_related"
    assert response.intent == "ticket_status"
    assert response.refused is False
    assert response.retrieval_chain == "ticket_augmented"
    assert response.business_snapshot["source"] == "business_adapter"
    assert response.business_snapshot["current_node"] == "DEPT_REVIEW"
    assert response.evidence_decision["reason"] == "business_status_verified"
    assert "部门审核中" in response.answer
    assert audit["business_status"]["current_node"] == "DEPT_REVIEW"
    assert "source_type" not in audit["retrieval_filter"]
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

    assert response.route == "ticket_context_related"
    assert response.intent == "rejection_reason"
    assert response.refused is False
    assert "身份证正面照片" in response.answer
    assert audit["reject_info"]["reject_code"] == "PHOTO_BLUR"
    assert audit["material_mapping"]["material_code"] == "ID_CARD_FRONT"
    assert "source_type" not in audit["retrieval_filter"]
    assert "material_code" not in audit["retrieval_filter"]
    assert "reject_code" not in audit["retrieval_filter"]
    assert response.business_snapshot["reject_code"] == "PHOTO_BLUR"
    assert response.retrieval_chain == "ticket_augmented"
    assert response.evidence_decision["ok"] is True


def test_reject_chain_ignores_conflicting_request_metadata_filter():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="为什么我的申请被退回了",
            user_id=1,
            region_code="340000",
            apply_id="A3003",
            metadata_filter={
                "bizType": "residence_permit",
                "materialCode": "RESIDENCE_PROOF",
                "rejectCode": "MISSING_RESIDENCE_PROOF",
            },
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.refused is False
    assert "biz_type" not in audit["retrieval_filter"]
    assert "material_code" not in audit["retrieval_filter"]
    assert "reject_code" not in audit["retrieval_filter"]
    assert "身份证正面照片" in response.answer


def test_reject_chain_refuses_when_business_reject_record_missing():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="为什么我的申请被退回了",
            user_id=1,
            region_code="340000",
            apply_id="A1001",
        )
    )

    assert response.intent == "rejection_reason"
    assert response.refused is True
    assert response.refuse_reason == "business_reject_not_found"
    assert "不能根据模型推测驳回原因" in response.answer


def test_reject_flow_falls_back_to_material_rule_without_reject_code():
    engine = InMemoryRAGEngine()
    engine.index_document(
        KnowledgeDocument(
            id="doc-id-card-material-rule",
            title="身份证正面照片材料规范",
            content="身份证正面照片不清晰时，需要重新上传边框完整、文字清楚、无遮挡、无反光的图片。",
            region_code="340000",
            policy_version="2025",
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

    assert response.intent == "rejection_reason"
    assert response.refused is False
    assert "source_type" not in audit["retrieval_filter"]
    assert "material_code" not in audit["retrieval_filter"]
    assert "reject_code" not in audit["retrieval_filter"]


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
    assert "audit_node" in sql
    assert "app_version" not in sql.lower()


def test_knowledge_chain_uses_full_corpus_without_metadata_filtering():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="办理居住证需要哪些材料",
            user_id=1,
            region_code="110000",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.route == "ticket_context_unrelated"
    assert response.intent == "material_requirements"
    assert response.refused is False
    assert audit["retrieval_filter"] == {}
    assert audit["rag_status"]["metadata_filtering"] is False
    assert audit["retrieved_sources"]
    assert "居住证" in response.answer


def test_rag_cache_reports_hit_on_repeated_query():
    engine = build_seed_rag_engine()
    filters = {
        "region_code": "340000",
        "policy_version": "2025",
        "status": "published",
    }

    first = engine.retrieve("办理居住证需要哪些材料", filters)
    second = engine.retrieve("办理居住证需要哪些材料", filters)

    assert first.status["cache"]["hit"] is False
    assert second.status["cache"]["hit"] is True
    assert second.status["cache"]["backend"] == "memory"
    assert second.status["vector_backend"] == "memory"


def test_rag_retrieval_degrades_to_bm25_when_dense_embedding_fails():
    class FailingEmbeddingProvider:
        name = "failing"
        dimension = 3

        def embed_query(self, text: str):
            raise TimeoutError("embedding timed out")

        def embed_documents(self, texts: list[str]):
            raise TimeoutError("embedding timed out")

    engine = InMemoryRAGEngine(embedding_provider=FailingEmbeddingProvider())
    document = KnowledgeDocument(id="doc-housing-fund", title="住房公积金提取", content="")
    chunk = KnowledgeChunk(
        id="chunk-housing-fund",
        doc_id=document.id,
        parent_id="",
        chunk_type="child",
        title=document.title,
        content="住房公积金提取需要身份证、银行卡、提取申请材料。",
    )
    engine.load_indexed_data([document], [chunk])

    result = engine.retrieve("住房公积金提取需要什么材料", {}, top_k=1)

    assert result.sources
    assert result.sources[0].chunk.id == "chunk-housing-fund"
    assert result.status["channels"]["dense"] == 0
    assert result.status["channels"]["bm25"] == 1
    assert result.status["degraded"] == [{"stage": "dense_retrieve", "error": "TimeoutError"}]


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
