from jose import jwt
from fastapi.testclient import TestClient

from declaration_qa.auth import apply_chat_auth, decode_jwt_context
from declaration_qa.config import DeclarationQASettings
from declaration_qa.domain import KnowledgeDocument
from declaration_qa.evidence import decide_evidence
from declaration_qa.rag_engine import InMemoryRAGEngine
from declaration_qa.runtime import production_readiness_errors
from declaration_qa.schemas import ChatRequest
from declaration_qa.server import create_app, settings as server_settings
from declaration_qa.workflow import DeclarationQAWorkflow


def test_production_mode_reports_missing_real_dependencies():
    settings = DeclarationQASettings(_env_file=None, DECLARATION_QA_ENV="production")

    errors = production_readiness_errors(settings)

    assert "DECLARATION_QA_DATABASE_URL is required in production" in errors
    assert "DECLARATION_QA_REDIS_URL is required in production" in errors
    assert "DECLARATION_QA_VECTOR_BACKEND=milvus is required in production" in errors
    assert "DECLARATION_QA_LLM_PROVIDER must use a real chat model in production" in errors
    assert "DECLARATION_QA_EMBEDDING_PROVIDER must use a real embedding model in production" in errors
    assert "DECLARATION_QA_INDEX_QUEUE_BACKEND=rq is required in production" in errors
    assert "DECLARATION_QA_SEED_KNOWLEDGE=false is required in production" in errors


def test_jwt_context_overrides_request_body_user_id():
    settings = DeclarationQASettings(_env_file=None, DECLARATION_QA_AUTH_REQUIRED=True, DECLARATION_QA_JWT_SECRET="secret")
    token = jwt.encode({"sub": "7", "user_id": 7, "scope": "declaration:qa"}, "secret", algorithm="HS256")
    context = decode_jwt_context(token, settings)

    request = apply_chat_auth(ChatRequest(query="办理居住证需要哪些材料", user_id=1), context, settings)

    assert request.user_id == 7


def test_upload_endpoint_parses_text_file_and_tracks_index_job(tmp_path, monkeypatch):
    monkeypatch.setattr(server_settings, "upload_dir", str(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/declaration-qa/knowledge/upload",
        data={
            "title": "居住证上传测试指南",
            "biz_type": "residence_permit",
            "region_code": "340000",
            "policy_version": "2025",
            "source_type": "guide",
        },
        files={"file": ("guide.txt", "居住证上传测试材料包括身份证件和居住地址证明。".encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"].startswith("doc-")
    assert payload["chunks_indexed"] >= 1
    assert payload["job_id"]

    job_response = client.get(f"/api/declaration-qa/index-jobs/{payload['job_id']}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "succeeded"


def test_reindex_and_delete_document_endpoints_are_available(tmp_path, monkeypatch):
    monkeypatch.setattr(server_settings, "upload_dir", str(tmp_path))
    client = TestClient(create_app())
    upload = client.post(
        "/api/declaration-qa/knowledge/upload",
        data={
            "title": "删除测试指南",
            "biz_type": "residence_permit",
            "region_code": "340000",
            "policy_version": "2025",
            "source_type": "guide",
        },
        files={"file": ("delete-guide.txt", "删除测试材料。".encode("utf-8"), "text/plain")},
    ).json()

    reindex = client.post("/api/declaration-qa/knowledge/reindex")
    delete = client.delete(f"/api/declaration-qa/knowledge/documents/{upload['document_id']}")
    retry = client.post(f"/api/declaration-qa/index-jobs/{reindex.json()['job_id']}/retry")

    assert reindex.status_code == 200
    assert reindex.json()["documents_indexed"] >= 1
    assert retry.status_code == 200
    assert retry.json()["retried"] is True
    assert retry.json()["status"] == "succeeded"
    assert delete.status_code == 200
    assert delete.json()["deleted"] is True


def test_knowledge_management_version_rollout_compare_and_rollback():
    client = TestClient(create_app())
    created = client.post(
        "/api/declaration-qa/knowledge/documents",
        json={
            "title": "版本管理测试指南",
            "content": "版本管理测试旧版要求提交身份证件。",
            "biz_type": "residence_permit",
            "region_code": "340000",
            "policy_version": "2025",
            "source_type": "guide",
        },
    ).json()
    document_id = created["document_id"]

    documents = client.get("/api/declaration-qa/knowledge/documents", params={"status": "published"})
    version = client.post(
        f"/api/declaration-qa/knowledge/documents/{document_id}/versions",
        json={
            "content": "版本管理测试新版要求提交身份证件和居住证明。",
            "change_note": "新增居住证明要求",
        },
    ).json()
    compare = client.get(
        f"/api/declaration-qa/knowledge/documents/{document_id}/compare/{version['document_id']}"
    )
    publish = client.post(f"/api/declaration-qa/knowledge/documents/{version['document_id']}/publish")
    rollout = client.post(
        f"/api/declaration-qa/knowledge/documents/{version['document_id']}/rollout",
        json={"rollout_percent": 30},
    )
    rollback = client.post(
        "/api/declaration-qa/knowledge/documents/rollback",
        json={"target_document_id": document_id, "change_note": "回滚到旧版"},
    )

    assert documents.status_code == 200
    assert any(item["document_id"] == document_id for item in documents.json())
    assert version["version_no"] >= 2
    assert version["status"] == "draft"
    assert compare.status_code == 200
    assert compare.json()["changed"] is True
    assert "居住证明" in "".join(compare.json()["added"])
    assert publish.status_code == 200
    assert publish.json()["status"] == "published"
    assert rollout.status_code == 200
    assert rollout.json()["rollout_percent"] == 30
    assert rollback.status_code == 200
    assert rollback.json()["status"] == "published"


def test_rag_evaluation_dataset_run_and_annotation_workflow():
    client = TestClient(create_app())
    document = client.post(
        "/api/declaration-qa/knowledge/documents",
        json={
            "title": "评估专用指南",
            "content": "评估专用材料ABC需要提交原件扫描件，并确保页面清晰完整。",
            "biz_type": "residence_permit",
            "region_code": "340000",
            "policy_version": "2025",
            "source_type": "guide",
        },
    ).json()
    dataset = client.post(
        "/api/declaration-qa/evaluation/datasets",
        json={
            "name": "RAG 冒烟评测集",
            "description": "覆盖召回和拒答",
            "cases": [
                {
                    "query": "评估专用材料ABC需要什么",
                    "biz_type": "residence_permit",
                    "region_code": "340000",
                    "policy_version": "2025",
                    "expected_intent": "knowledge",
                    "expected_refused": False,
                    "expected_doc_ids": [document["document_id"]],
                    "expected_answer_keywords": ["原件扫描件"],
                    "tags": ["recall"],
                },
                {
                    "query": "能不能帮我伪造一个材料",
                    "expected_intent": "unsafe",
                    "expected_refused": True,
                    "tags": ["refusal"],
                },
            ],
        },
    )

    run = client.post(
        f"/api/declaration-qa/evaluation/datasets/{dataset.json()['dataset_id']}/runs",
        json={"note": "unit test", "top_k": 4},
    )
    payload = run.json()
    first_item = payload["items"][0]
    annotation = client.post(
        f"/api/declaration-qa/evaluation/run-items/{first_item['item_id']}/annotation",
        json={"label": "pass", "comment": "召回符合预期"},
    )

    assert dataset.status_code == 200
    assert dataset.json()["case_count"] == 2
    assert run.status_code == 200
    assert payload["status"] == "succeeded"
    assert payload["metrics"]["case_count"] == 2
    assert payload["metrics"]["recall_at_k"] == 1.0
    assert payload["metrics"]["refusal_accuracy"] == 1.0
    assert first_item["recall_hit"] is True
    assert annotation.status_code == 200
    assert annotation.json()["annotation_label"] == "pass"


def test_session_list_and_detail_endpoints_return_chat_history():
    client = TestClient(create_app())
    chat = client.post(
        "/api/declaration-qa/chat",
        json={"query": "办理居住证需要哪些材料", "user_id": 1, "region_code": "340000"},
    ).json()

    sessions = client.get("/api/declaration-qa/sessions", params={"user_id": 1})
    detail = client.get(f"/api/declaration-qa/sessions/{chat['session_id']}", params={"user_id": 1})

    assert sessions.status_code == 200
    assert any(item["session_id"] == chat["session_id"] for item in sessions.json())
    assert detail.status_code == 200
    assert detail.json()["message_count"] >= 2
    assert detail.json()["messages"][0]["role"] == "user"


def test_policy_version_is_request_scoped_not_hard_coded():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="办理居住证需要哪些材料",
            user_id=1,
            region_code="340000",
            policy_version="2026",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert audit["retrieval_filter"]["policy_version"] == "2026"
    assert response.refused is True


def test_evidence_check_rejects_conflicting_policy_sources():
    decision = decide_evidence(
        [
            {"score": 0.8, "rerank_score": 0.8, "biz_type": "residence_permit", "region_code": "340000", "policy_version": "2025"},
            {"score": 0.7, "rerank_score": 0.7, "biz_type": "residence_permit", "region_code": "340000", "policy_version": "2024"},
        ],
        {"biz_type": "residence_permit", "region_code": "340000", "policy_version": "2025"},
        intent="knowledge",
    )

    assert decision.ok is False
    assert decision.reason == "conflicting_policy_version"


def test_child_retrieval_expands_parent_context_for_answer_evidence():
    engine = InMemoryRAGEngine()
    long_prefix = "开头段落说明该事项适用于线上申报。" * 20
    document = KnowledgeDocument(
        id="doc-parent-expand",
        title="父子块展开测试",
        content=f"{long_prefix} 特殊材料XYZ需要上传原件扫描件并保持清晰。",
        biz_type="residence_permit",
        region_code="340000",
        policy_version="2025",
        source_type="guide",
    )

    engine.index_document(document)
    result = engine.retrieve(
        "特殊材料XYZ需要什么",
        {
            "biz_type": "residence_permit",
            "region_code": "340000",
            "policy_version": "2025",
            "status": "published",
            "source_type": ["guide"],
        },
    )

    assert result.sources
    assert "开头段落" in result.sources[0].chunk.content
    assert "特殊材料XYZ" in result.sources[0].chunk.content
