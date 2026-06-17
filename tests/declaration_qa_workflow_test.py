from fastapi.testclient import TestClient
from pathlib import Path

from declaration_qa.business import InMemoryBusinessDataAdapter
from declaration_qa.domain import KnowledgeChunk, KnowledgeDocument
from declaration_qa.prompting import MULTI_QUESTION_ANSWER_PROMPT, format_sub_task_payload
from declaration_qa.question_planner import LLMQuestionPlanner, QUESTION_PLANNER_PROMPT, RuleQuestionPlanner
from declaration_qa.rag_engine import InMemoryRAGEngine, build_seed_rag_engine
from declaration_qa.repository import PostgresDeclarationQAStore
from declaration_qa.router import parse_json_object
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


def test_rule_question_planner_splits_multi_question():
    planner = RuleQuestionPlanner()

    plan = planner.plan(
        {
            "query": "办理居住证需要哪些材料？多久能办完，收费吗？",
            "normalized_query": "办理居住证需要哪些材料？多久能办完，收费吗？",
            "region_code": "340000",
        }
    )

    assert plan.is_multi_question is True
    assert [item["query"] for item in plan.questions] == ["办理居住证需要哪些材料", "多久能办完，收费吗"]


def test_rule_question_planner_keeps_single_question_in_questions():
    planner = RuleQuestionPlanner()

    plan = planner.plan(
        {
            "query": "办理健康证需要哪些材料？",
            "normalized_query": "办理健康证需要哪些材料？",
            "region_code": "340000",
        }
    )

    assert plan.is_multi_question is False
    assert [item["query"] for item in plan.questions] == ["办理健康证需要哪些材料"]


def test_question_planner_prompt_limits_over_splitting_and_markdown():
    assert "如果不是多问题，也要返回 questions 数组" in QUESTION_PLANNER_PROMPT
    assert "只输出合法 JSON，不要输出 markdown" in QUESTION_PLANNER_PROMPT
    assert "同一个业务问题中的限定条件" in QUESTION_PLANNER_PROMPT
    assert "不输出 intent" in QUESTION_PLANNER_PROMPT
    assert "不输出 route" in QUESTION_PLANNER_PROMPT
    assert "示例事项甲" in QUESTION_PLANNER_PROMPT
    assert "示例证照乙" in QUESTION_PLANNER_PROMPT
    assert "防火办事流程" not in QUESTION_PLANNER_PROMPT
    assert "草原防火期" not in QUESTION_PLANNER_PROMPT
    assert "健康证" not in QUESTION_PLANNER_PROMPT


def test_llm_question_planner_keeps_only_understanding_fields():
    class FakeLLM:
        def invoke(self, prompt):
            return """
            {
              "is_multi_question": false,
              "questions": [
                {
                  "original_query": "防火办事流程",
                  "query": "草原防火期内在草原上进行爆破、勘察和施工等活动审批的办理流程是什么？",
                  "requires_ticket_context": false,
                  "is_expanded": true,
                  "expansion_reason": "keyword_query_to_process_question",
                  "intent": "process_and_time",
                  "route": "ticket_context_unrelated"
                }
              ],
              "reason": "keyword_query_expanded"
            }
            """

    plan = LLMQuestionPlanner(FakeLLM()).plan({"query": "防火办事流程", "normalized_query": "防火办事流程"})

    assert plan.is_multi_question is False
    assert plan.questions[0]["original_query"] == "防火办事流程"
    assert plan.questions[0]["query"] == "草原防火期内在草原上进行爆破、勘察和施工等活动审批的办理流程是什么"
    assert plan.questions[0]["is_expanded"] is True
    assert "intent" not in plan.questions[0]
    assert "route" not in plan.questions[0]


def test_llm_question_planner_keeps_ticket_related_query_unexpanded():
    class FakeLLM:
        def invoke(self, prompt):
            return """
            {
              "is_multi_question": false,
              "questions": [
                {
                  "original_query": "这个要哪些材料",
                  "query": "涉外收养登记需要哪些材料",
                  "requires_ticket_context": true,
                  "is_expanded": true,
                  "expansion_reason": "expanded_with_current_ticket_title"
                }
              ],
              "reason": "ticket_related_query"
            }
            """

    plan = LLMQuestionPlanner(FakeLLM()).plan(
        {
            "query": "这个要哪些材料",
            "normalized_query": "这个要哪些材料",
            "apply_id": "AHZWFW-REAL40-MOCK-001",
            "apply_context": {
                "apply_id": "AHZWFW-REAL40-MOCK-001",
                "title": "涉外收养登记",
                "biz_type": "foreign_related_adoption_registration",
            },
        }
    )

    assert plan.questions[0]["query"] == "这个要哪些材料"
    assert plan.questions[0]["requires_ticket_context"] is True
    assert plan.questions[0]["is_expanded"] is False
    assert plan.questions[0]["expansion_reason"] == "ticket_related_query_kept_unexpanded"


def test_single_question_workflow_uses_planner_expanded_query_without_planner_route():
    class ExpandingPlanner:
        def plan(self, state):
            return type(
                "Plan",
                (),
                {
                    "is_multi_question": False,
                    "questions": [
                        {
                            "id": "q1",
                            "original_query": "防火办事流程",
                            "query": "草原防火期内在草原上进行爆破、勘察和施工等活动审批的办理流程是什么",
                            "requires_ticket_context": False,
                            "is_expanded": True,
                            "expansion_reason": "keyword_query_to_process_question",
                        }
                    ],
                    "reason": "keyword_query_expanded",
                    "provider": "test",
                    "payload": lambda self: {
                        "is_multi_question": self.is_multi_question,
                        "questions": self.questions,
                        "reason": self.reason,
                        "provider": self.provider,
                    },
                },
            )()

    workflow = DeclarationQAWorkflow(question_planner=ExpandingPlanner())
    response = workflow.invoke(ChatRequest(query="防火办事流程", user_id=1, region_code="340000"))

    assert response.multi_question is False
    assert response.question_plan["questions"][0]["original_query"] == "防火办事流程"
    assert response.question_plan["questions"][0]["is_expanded"] is True
    assert response.route in {"ticket_context_unrelated", "clarification", "refuse"}
    assert response.route_decision.get("route") == response.route
    question_plan_event = next(event for event in response.events if event["event"] == "question_plan")
    assert question_plan_event["data"]["expanded"] is True


def test_parse_json_object_strips_uppercase_json_markdown_fence():
    payload = parse_json_object('```JSON\n{"is_multi_question": false, "questions": []}\n```')

    assert payload == {"is_multi_question": False, "questions": []}


def test_multi_question_prompt_uses_shared_evidence_pool_and_refs():
    source_a = {"chunk_id": "doc-a:chunk-1", "doc_id": "doc-a", "title": "指南A", "snippet": "重复证据"}
    payload = format_sub_task_payload(
        [
            {"query": "问题一", "route": "ticket_context_unrelated", "sources": [source_a]},
            {"query": "问题二", "route": "ticket_context_unrelated", "sources": [source_a]},
        ]
    )

    assert "<SharedEvidencePool>" in payload
    assert payload.count("<Evidence id=\"chunk_") == 1
    assert payload.count("<Ref>chunk_") == 2
    assert payload.count("重复证据") == 1


def test_multi_question_workflow_groups_sub_question_evidence_for_one_final_answer():
    class CapturingAnswerGenerator:
        def generate(self, state):
            self.state = dict(state)
            prompt = MULTI_QUESTION_ANSWER_PROMPT.format(
                query=state.get("query") or "",
                sub_task_payload=format_sub_task_payload(state.get("sub_tasks") or []),
            )
            return "多问题最终回答", prompt

    answer_generator = CapturingAnswerGenerator()
    workflow = DeclarationQAWorkflow(answer_generator=answer_generator)

    response = workflow.invoke(
        ChatRequest(
            query="我的健康证办理到哪一步了？办理居住证需要哪些材料？",
            user_id=1,
            region_code="340000",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.multi_question is True
    assert response.route == "multi_question"
    assert response.intent == "multi_question"
    assert response.refused is False
    assert response.answer == "多问题最终回答"
    assert [task["query"] for task in response.sub_tasks] == ["我的健康证办理到哪一步了", "办理居住证需要哪些材料"]
    assert response.sub_tasks[0]["route"] == "ticket_context_related"
    assert response.sub_tasks[0]["intent"] == "ticket_status"
    assert response.sub_tasks[0]["status"] == "ready"
    assert response.sub_tasks[0]["business_snapshot"]["current_node"] == "DEPT_REVIEW"
    assert response.sub_tasks[1]["route"] == "ticket_context_unrelated"
    assert response.sub_tasks[1]["intent"] == "material_requirements"
    assert response.sub_tasks[1]["status"] == "ready"
    assert "问题1" in audit["final_prompt"]
    assert "我的健康证办理到哪一步了" in audit["final_prompt"]
    assert "问题2" in audit["final_prompt"]
    assert "办理居住证需要哪些材料" in audit["final_prompt"]
    assert audit["multi_question"] is True
    assert len(audit["sub_tasks"]) == 2


def test_multi_question_requires_ticket_selection_before_answer_generation():
    class FailIfCalledAnswerGenerator:
        def generate(self, state):
            raise AssertionError("multi-question ticket selection should happen before final answer generation")

    workflow = DeclarationQAWorkflow(answer_generator=FailIfCalledAnswerGenerator())

    response = workflow.invoke(ChatRequest(query="我的申请办到哪了？办理居住证需要哪些材料？", user_id=1, region_code="340000"))

    assert response.multi_question is True
    assert response.need_ticket_selection is True
    assert response.ticket_selection_reason == "multiple_candidates"
    assert response.refused is True
    assert response.refuse_reason == "ticket_selection_required"
    assert response.sub_tasks[0]["status"] == "need_ticket_selection"
    assert response.sub_tasks[0]["need_ticket_selection"] is True
    assert len(response.sub_tasks[0]["tickets"]) >= 2
    assert response.sub_tasks[1]["status"] == "ready"
    assert response.sub_tasks[1]["route"] == "ticket_context_unrelated"
    assert response.sources == []
    assert "多个可能相关的申请单" in response.answer


def test_material_definition_question_uses_current_ticket_context():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="申请报告是干嘛的",
            user_id=1,
            region_code="340000",
            apply_id="A1001",
        )
    )

    assert response.route == "ticket_context_related"
    assert response.intent == "knowledge_qa"
    assert response.need_ticket_selection is False
    assert response.ticket_selection_reason == "provided_apply_context"
    assert response.retrieval_chain == "ticket_augmented"
    assert response.ticket_context["ticket"]["ticket_id"] == "A1001"


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


def test_question_planner_drives_ticket_context_route_without_router():
    class TicketRelatedPlanner:
        def plan(self, state):
            return type(
                "Plan",
                (),
                {
                    "is_multi_question": False,
                    "questions": [
                        {
                            "id": "q1",
                            "original_query": "麻烦看一下这单现在什么情况",
                            "query": "麻烦看一下这单现在什么情况",
                            "requires_ticket_context": True,
                            "is_expanded": False,
                            "expansion_reason": "",
                        }
                    ],
                    "reason": "planner_ticket_related",
                    "provider": "test",
                    "payload": lambda self: {
                        "is_multi_question": self.is_multi_question,
                        "questions": self.questions,
                        "reason": self.reason,
                        "provider": self.provider,
                    },
                },
            )()

    workflow = DeclarationQAWorkflow(question_planner=TicketRelatedPlanner())

    response = workflow.invoke(
        ChatRequest(query="麻烦看一下这单现在什么情况", user_id=1, region_code="340000", apply_id="A1001")
    )

    assert response.route == "ticket_context_related"
    assert response.intent == "ticket_status"
    assert response.route_decision["entities"]["router_provider"] == "question_planner"
    assert "部门审核中" in response.answer


def test_question_planner_drives_knowledge_route_without_router():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(ChatRequest(query="办理居住证需要哪些材料", user_id=1, region_code="340000"))

    assert response.route == "ticket_context_unrelated"
    assert response.intent == "material_requirements"
    assert response.route_decision["entities"]["router_provider"] == "question_planner"


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


def test_multi_question_stream_requires_ticket_selection_before_tokens():
    client = TestClient(create_app())

    response = client.post(
        "/api/declaration-qa/chat/stream",
        json={
            "query": "我的申请办到哪了？办理居住证需要哪些材料？",
            "user_id": 1,
            "region_code": "340000",
        },
    )

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "多个可能相关的申请单" in response.text
    assert "need_ticket_selection" in response.text
    assert "办理居住证需要准备" not in response.text
    assert "event: sources" not in response.text


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
