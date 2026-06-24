from jose import jwt
from fastapi.testclient import TestClient

from declaration_qa.api.auth import apply_chat_auth, decode_jwt_context
from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.core.domain import KnowledgeDocument
from declaration_qa.rag.evidence import decide_evidence
from declaration_qa.indexing.service import extract_document_metadata
from declaration_qa.rag.engine import InMemoryRAGEngine, split_document
from declaration_qa.core.runtime import production_readiness_errors
from declaration_qa.api.schemas import ChatRequest
from declaration_qa.api.server import create_app, settings as server_settings, stable_upload_document_id
from declaration_qa.application.workflow import DeclarationQAWorkflow


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


def test_build_query_rewriter_uses_dedicated_llm_model(monkeypatch):
    import declaration_qa.rag.model_providers as providers

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "langchain_openai", type("M", (), {"ChatOpenAI": FakeChatOpenAI}))

    rewriter = providers.build_query_rewriter(
        DeclarationQASettings(
            _env_file=None,
            DECLARATION_QA_LLM_PROVIDER="openai",
            DECLARATION_QA_OPENAI_API_KEY="key",
            DECLARATION_QA_LLM_MODEL="general-model",
            DECLARATION_QA_QUERY_REWRITER_MODEL="rewrite-model",
        )
    )

    assert rewriter.name == "llm"
    assert captured["model"] == "rewrite-model"


def test_upload_endpoint_parses_text_file_and_tracks_index_job(tmp_path, monkeypatch):
    monkeypatch.setattr(server_settings, "upload_dir", str(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/declaration-qa/knowledge/upload",
        data={
            "title": "居住证上传测试指南",
            "region_code": "340000",
            "policy_version": "2025",
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


def test_upload_document_id_is_stable_for_same_file_content():
    first = stable_upload_document_id("重复上传测试内容".encode("utf-8"))
    second = stable_upload_document_id("重复上传测试内容".encode("utf-8"))

    assert first == second
    assert first.startswith("doc-")
    assert len(first) == 36


def test_upload_metadata_extractor_reads_real_government_document_fields():
    content = """
# 涉外收养登记流程文档

- Region Code: 340000000000
- Region Name: 安徽省
- Policy Version: ahzwfw-2026-06-15
- Official Source URL: https://www.ahzwfw.gov.cn/bog-bsdt/static/workProcess.html?ssqdId=abc123&ssqdCode=113400000000001234&typeSipe=nameSkip

## 基本信息

| 字段 | 内容 |
| --- | --- |
| 主题分类 | 生育收养 |
| 主题编码 | A12 |
| 目录清单名称 | 涉外收养登记 |
| 目录子项名称 | 无 |
| 事项名称 | 涉外收养登记 |
| 事项编码 | 113400000000009999 |
| 事项类型 | 行政确认 |
| 服务对象 | 自然人 |
| 实施主体 | 安徽省民政厅 |

## 申请材料
居民户口簿、身份证件、收养登记申请书。
"""

    metadata = extract_document_metadata(content, title="涉外收养登记流程文档")

    assert metadata["region_key"] == "340000000000"
    assert metadata["region_name"] == "安徽省"
    assert metadata["catalog_name"] == "涉外收养登记"
    assert metadata["catalog_subitem_name"] == "无"
    assert metadata["item_code"] == "113400000000009999"
    assert metadata["item_name"] == "涉外收养登记"
    assert metadata["item_type"] == "行政确认"
    assert metadata["theme_code"] == "A12"
    assert "theme_name" not in metadata
    assert metadata["service_object"] == "自然人"
    assert metadata["source_version"] == "ahzwfw-2026-06-15"
    assert metadata["official_source_url"].startswith("https://www.ahzwfw.gov.cn/")
    assert metadata["ssqd_code"] == "113400000000001234"
    assert metadata["ssqd_id"] == "abc123"


def test_reindex_and_delete_document_endpoints_are_available(tmp_path, monkeypatch):
    monkeypatch.setattr(server_settings, "upload_dir", str(tmp_path))
    client = TestClient(create_app())
    upload = client.post(
        "/api/declaration-qa/knowledge/upload",
        data={
            "title": "删除测试指南",
            "region_code": "340000",
            "policy_version": "2025",
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
            "region_code": "340000",
            "policy_version": "2025",
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
            "region_code": "340000",
            "policy_version": "2025",
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
                    "expected_intent": "material_requirements",
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


def test_delete_session_endpoint_removes_chat_history():
    client = TestClient(create_app())
    chat = client.post(
        "/api/declaration-qa/chat",
        json={"query": "删除会话测试", "user_id": 1, "region_code": "340000"},
    ).json()

    deleted = client.delete(f"/api/declaration-qa/sessions/{chat['session_id']}", params={"user_id": 1})
    sessions = client.get("/api/declaration-qa/sessions", params={"user_id": 1})
    detail = client.get(f"/api/declaration-qa/sessions/{chat['session_id']}", params={"user_id": 1})

    assert deleted.status_code == 200
    assert deleted.json() == {"session_id": chat["session_id"], "deleted": True}
    assert all(item["session_id"] != chat["session_id"] for item in sessions.json())
    assert detail.status_code == 404


def test_policy_version_is_retained_but_not_used_as_retrieval_filter():
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

    assert audit["retrieval_filter"] == {}
    assert audit["rag_status"]["metadata_filtering"] is False
    assert response.refused is False


def test_evidence_check_does_not_reject_conflicting_metadata_versions():
    decision = decide_evidence(
        [
            {
                "doc_id": "doc-2025",
                "title": "2025 事项流程文档",
                "score": 0.8,
                "rerank_score": 0.8,
                "region_code": "340000",
                "policy_version": "2025",
            },
            {
                "doc_id": "doc-2024",
                "title": "2024 事项流程文档",
                "score": 0.7,
                "rerank_score": 0.7,
                "region_code": "340000",
                "policy_version": "2024",
            },
        ],
        {"region_code": "340000", "policy_version": "2025"},
        intent="material_requirements",
    )

    assert decision.ok is True
    assert decision.reason == ""


def test_child_retrieval_expands_parent_context_for_answer_evidence():
    engine = InMemoryRAGEngine()
    long_prefix = "开头段落说明该事项适用于线上申报。" * 20
    document = KnowledgeDocument(
        id="doc-parent-expand",
        title="父子块展开测试",
        content=f"{long_prefix} 特殊材料XYZ需要上传原件扫描件并保持清晰。",
        region_code="340000",
        policy_version="2025",
    )

    engine.index_document(document)
    result = engine.retrieve(
        "特殊材料XYZ需要什么",
        {
            "region_code": "340000",
            "policy_version": "2025",
            "status": "published",
        },
    )

    assert result.sources
    assert "开头段落" in result.sources[0].chunk.content
    assert "特殊材料XYZ" in result.sources[0].chunk.content


def test_hierarchical_chunking_preserves_markdown_heading_path():
    document = KnowledgeDocument(
        id="doc-hierarchical",
        title="层级分块测试",
        content="# 办事指南\n\n## 申请材料\n\n### 1. 身份证件\n| 字段 | 内容 |\n| --- | --- |\n| 材料必要性 | 告知承诺 |\n\n### 2. 申请表\n| 字段 | 内容 |\n| --- | --- |\n| 材料必要性 | 必要 |\n\n## 办理流程\n\n### 办理环节\n\n#### 受理\n| 字段 | 内容 |\n| --- | --- |\n| 办理时限 | 1个工作日 |",
        metadata={"chunking_strategy": "hierarchical"},
    )

    chunks = split_document(document)

    assert chunks
    assert all(chunk.chunk_type == "child" for chunk in chunks)
    assert not any(chunk.content == "办事指南 > 申请材料" for chunk in chunks)
    assert not any(chunk.content == "办事指南 > 办理流程" for chunk in chunks)
    assert len({chunk.content for chunk in chunks}) == len(chunks)
    material = next(chunk for chunk in chunks if chunk.metadata.get("node_title") == "1. 身份证件")
    assert material.metadata["heading_path"] == "办事指南 > 申请材料 > 1. 身份证件"
    assert material.metadata["section_title"] == "申请材料"
    assert material.metadata["parent_section_id"] == "doc-hierarchical#申请材料"
    assert material.metadata["node_order"] == 1
    assert material.metadata["chunk_type"] == "material_item"
    assert material.metadata["indexable"] is True
    assert "| 字段 | 内容 |\n| --- | --- |\n| 材料必要性 | 告知承诺 |" in material.content
    process = next(chunk for chunk in chunks if chunk.metadata.get("node_title") == "受理")
    assert process.metadata["section_title"] == "办理流程"
    assert process.metadata["chunk_type"] == "process_step"
    assert all(chunk.metadata.get("chunking_strategy") == "hierarchical" for chunk in chunks)


def test_hierarchical_chunking_uses_context_parent_only_for_long_sections():
    document = KnowledgeDocument(
        id="doc-hierarchical-long",
        title="长层级分块测试",
        content="# 办事指南\n\n## 申请材料\n\n### 1. 申请表\n" + "\n".join(f"| 字段{i} | 内容{i}需要保持表格换行 |" for i in range(40)),
        metadata={"chunking_strategy": "hierarchical"},
    )

    chunks = split_document(document)
    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]

    assert parents
    assert children
    assert all(parent.metadata.get("context_only") is True for parent in parents)
    assert all(parent.metadata.get("indexable") is False for parent in parents)
    assert not any(parent.content == child.content for parent in parents for child in children)
    assert all("\n" in child.content for child in children)
    assert {chunk.metadata.get("parent_section_id") for chunk in chunks} == {"doc-hierarchical-long#申请材料"}


def test_material_query_backfills_sibling_items_by_parent_section_id():
    engine = InMemoryRAGEngine()
    document = KnowledgeDocument(
        id="doc-material-backfill",
        title="材料回填测试",
        content=(
            "# 办事指南\n\n"
            "## 申请材料\n\n"
            "### 1. 身份证件\n身份证件通过共享核验。\n\n"
            "### 2. 单位组织机构代码证\n单位组织机构代码证通过共享核验。\n\n"
            "### 3. 申请表\n申请表需要申请人自备。"
        ),
        metadata={"chunking_strategy": "hierarchical"},
    )

    engine.index_document(document)
    result = engine.retrieve("申请表需要哪些材料", {}, top_k=1)

    material_titles = [
        source.chunk.metadata.get("node_title")
        for source in result.sources
        if source.chunk.metadata.get("chunk_type") == "material_item"
    ]
    assert material_titles[:3] == ["1. 身份证件", "2. 单位组织机构代码证", "3. 申请表"]
    assert result.status["structured_backfill"]["applied"] is True
    assert result.status["structured_backfill"]["target_chunk_type"] == "material_item"


def test_material_backfill_merges_all_windows_for_long_item():
    engine = InMemoryRAGEngine()
    long_material_body = "\n".join(
        [f"| 字段{i} | 长材料内容{i}，用于制造多个 child window |" for i in range(30)]
        + ["| 结尾字段 | 长材料尾段XYZ必须保留 |"]
    )
    document = KnowledgeDocument(
        id="doc-material-long-backfill",
        title="长材料回填测试",
        content=(
            "# 办事指南\n\n"
            "## 申请材料\n\n"
            "### 1. 身份证件\n身份证件通过共享核验。\n\n"
            f"### 2. 长材料\n{long_material_body}"
        ),
        metadata={"chunking_strategy": "hierarchical"},
    )

    engine.index_document(document)
    result = engine.retrieve("长材料尾段XYZ需要哪些材料", {}, top_k=1)

    long_material = next(
        source.chunk
        for source in result.sources
        if source.chunk.metadata.get("node_title") == "2. 长材料"
    )
    assert long_material.metadata["merged_chunk_count"] > 1
    assert "字段0" in long_material.content
    assert "长材料尾段XYZ必须保留" in long_material.content
    assert len([
        source
        for source in result.sources
        if source.chunk.metadata.get("node_title") == "2. 长材料"
    ]) == 1


def test_process_query_backfills_sibling_steps_by_parent_section_id():
    engine = InMemoryRAGEngine()
    document = KnowledgeDocument(
        id="doc-process-backfill",
        title="流程回填测试",
        content=(
            "# 办事指南\n\n"
            "## 办理流程\n\n"
            "#### 受理\n办理时限：1个工作日。\n\n"
            "#### 审查\n办理时限：3个工作日。\n\n"
            "#### 决定\n办理时限：4个工作日。\n\n"
            "#### 办结\n办理时限：2个工作日。\n\n"
            "#### 送达\n办理时限：0个工作日。"
        ),
        metadata={"chunking_strategy": "hierarchical"},
    )

    engine.index_document(document)
    result = engine.retrieve("审查流程是啥", {}, top_k=1)

    process_titles = [
        source.chunk.metadata.get("node_title")
        for source in result.sources
        if source.chunk.metadata.get("chunk_type") == "process_step"
    ]
    assert process_titles[:5] == ["受理", "审查", "决定", "办结", "送达"]
    assert result.status["structured_backfill"]["applied"] is True
    assert result.status["structured_backfill"]["target_chunk_type"] == "process_step"
