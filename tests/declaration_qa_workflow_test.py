from fastapi.testclient import TestClient
from pathlib import Path
import time

from declaration_qa.core.business import InMemoryBusinessDataAdapter
from declaration_qa.core.domain import KnowledgeChunk, KnowledgeDocument, RetrievalResult, RetrievedSource
from declaration_qa.rag.prompting import MULTI_QUESTION_ANSWER_PROMPT, format_sub_task_payload
from declaration_qa.rag.query_rewriter.llm import LLMQueryRewriter
from declaration_qa.rag.query_rewriter.prompts import build_query_rewriter_prompt
from declaration_qa.rag.query_rewriter.rule import RuleQueryRewriter
from declaration_qa.rag.question_resolver.llm import LLMQuestionResolver
from declaration_qa.rag.question_resolver.prompts import QUESTION_RESOLVER_PROMPT
from declaration_qa.rag.question_resolver.rule import RuleQuestionResolver
from declaration_qa.rag.engine.in_memory import InMemoryRAGEngine
from declaration_qa.rag.engine.seed import build_seed_rag_engine
from declaration_qa.storage.repository.business import BusinessRepository
from declaration_qa.storage.repository.store import PostgresDeclarationQAStore
from declaration_qa.rag.router import parse_json_object
from declaration_qa.api.schemas import ChatImagePayload, ChatRequest
from declaration_qa.api.server.app import create_app
from declaration_qa.application.workflow import STREAM_TOKEN_DELAY_SECONDS, STREAM_TOKEN_SIZE, DeclarationQAWorkflow, select_answer_sources, ticket_rag_query, token_events


def test_declaration_qa_entrypoint_isolated_from_existing_main():
    client = TestClient(create_app())

    response = client.get("/api/declaration-qa/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["entrypoint"] == "declaration_qa.api.server.app:app"
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
    from declaration_qa.api.server import knowledge_admin

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
    monkeypatch.setattr(knowledge_admin, "workflow", fake_workflow)
    monkeypatch.setattr(knowledge_admin, "_knowledge_base_revision", "rev-old")

    knowledge_admin.ensure_knowledge_base_fresh()

    assert fake_workflow.rag_engine.loaded == ([document], [chunk])
    assert knowledge_admin._knowledge_base_revision == "rev-new"


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


def test_workflow_stream_uses_langgraph_custom_tokens_for_llm_node():
    class FakeStreamingGenerator:
        def stream_generate(self, state):
            return iter(["图", "流", "式"]), "prompt-text"

    workflow = DeclarationQAWorkflow(answer_generator=FakeStreamingGenerator())

    events = list(
        workflow.stream(
            ChatRequest(
                query="办理居住证需要哪些材料",
                user_id=1,
                region_code="340000",
            )
        )
    )

    assert [event["data"]["content"] for event in events if event["event"] == "token"] == ["图", "流", "式"]
    assert {"event": "answer_generated", "data": {"intent": "material_requirements", "streamed": True}} in events
    assert events[-1]["event"] == "done"
    audit = workflow.audit_store.get(events[-1]["data"]["audit_id"])
    assert audit["answer"] == "图流式"
    assert audit["final_prompt"] == "prompt-text"


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


def test_query_rewriter_scopes_unrelated_query_without_ticket_facts():
    rewriter = RuleQueryRewriter()

    rewrite = rewriter.rewrite(
        {
            "query": "这个事情最快几天能拿结果",
            "normalized_query": "这个事情最快几天能拿结果",
            "requires_ticket_context": False,
            "ticket_fact_type": "none",
            "retrieval_plan": {
                "metadata_needs": ["item_scope"],
                "ticket_fact_type": "none",
            },
            "apply_context": {
                "biz_type": "grassland_fire_blasting_approval",
                "reject_reason": "申请表缺少防火措施说明",
                "payload": {
                    "official_item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批",
                },
            },
        }
    )

    assert len(rewrite.queries) == 3
    assert all("草原防火期内在草原上进行爆破、勘察和施工等活动审批" in item.text for item in rewrite.queries)
    assert all("申请表缺少防火措施说明" not in item.text for item in rewrite.queries)
    assert any("法定办结时限" in item.text for item in rewrite.queries)


def test_query_rewriter_allows_ticket_facts_for_rejection_query():
    rewriter = RuleQueryRewriter()

    rewrite = rewriter.rewrite(
        {
            "query": "为什么被退回了",
            "normalized_query": "为什么被退回了",
            "requires_ticket_context": True,
            "ticket_fact_type": "rejection",
            "apply_context": {
                "biz_type": "grassland_fire_blasting_approval",
                "payload": {
                    "official_item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批",
                },
            },
            "reject_info": {
                "reject_reason": "申请表缺少防火措施说明",
                "material_name": "申请表",
            },
        }
    )

    assert rewrite.fact_text == "申请表缺少防火措施说明 申请表"
    assert all("申请表缺少防火措施说明" in item.text for item in rewrite.queries)


def test_query_rewriter_without_complete_sections_uses_scoped_original_only():
    rewrite = RuleQueryRewriter().rewrite(
        {
            "query": "这是什么意思",
            "normalized_query": "这是什么意思",
            "requires_ticket_context": False,
            "ticket_fact_type": "none",
            "apply_context": {
                "payload": {"official_item_name": "示例事项A"},
            },
        }
    )

    assert len(rewrite.queries) == 1
    assert rewrite.queries[0].strategy == "scoped_original"
    assert rewrite.queries[0].text == "示例事项A 这是什么意思"


def test_question_resolver_does_not_request_complete_sections():
    resolution = RuleQuestionResolver().resolve(
        {
            "query": "防火期在草原施工需要办理什么",
            "normalized_query": "防火期在草原施工需要办理什么",
        }
    )
    retrieval_plan = resolution.questions[0]["retrieval_plan"]

    assert "need_complete_sections" not in retrieval_plan
    assert retrieval_plan["ticket_fact_type"] == "none"


def test_llm_query_rewriter_uses_model_queries():
    class FakeLLM:
        def invoke(self, prompt):
            self.prompt = prompt
            return """
            {
              "queries": [
                {"text": "申请材料 材料清单 示例事项A", "strategy": "semantic_rewrite", "top_k": 4},
                {"text": "必要材料 原件 复印件 示例事项A", "strategy": "focus_terms", "top_k": 3}
              ],
              "reason": "material_query_rewrite"
            }
            """

    llm = FakeLLM()
    rewrite = LLMQueryRewriter(llm).rewrite(
        {
            "query": "要哪些材料",
            "normalized_query": "要哪些材料",
            "requires_ticket_context": False,
            "apply_context": {"payload": {"official_item_name": "示例事项A"}},
        }
    )

    assert rewrite.provider == "llm"
    assert rewrite.reason == "material_query_rewrite"
    assert [item.text for item in rewrite.queries] == ["申请材料 材料清单 示例事项A", "必要材料 原件 复印件 示例事项A"]
    assert rewrite.queries[0].top_k == 4
    assert "metadata_filter_has_item_scope" in llm.prompt


def test_llm_query_rewriter_falls_back_to_rule_on_invalid_json():
    class BadLLM:
        def invoke(self, prompt):
            return "not-json"

    rewrite = LLMQueryRewriter(BadLLM()).rewrite(
        {
            "query": "要哪些材料",
            "normalized_query": "要哪些材料",
            "requires_ticket_context": False,
            "apply_context": {"payload": {"official_item_name": "示例事项A"}},
        }
    )

    assert rewrite.provider == "rule_fallback"
    assert rewrite.queries
    assert rewrite.reason.startswith("llm_query_rewriter_fallback")


def test_query_rewriter_prompt_hides_scope_when_item_metadata_filter_exists():
    prompt = build_query_rewriter_prompt(
        {
            "query": "这个事情最快几天能拿结果",
            "normalized_query": "这个事情最快几天能拿结果",
            "requires_ticket_context": False,
            "retrieval_filter": {"item_code": "ITEM001"},
            "apply_context": {"payload": {"official_item_name": "示例事项A"}},
        }
    )

    assert '"metadata_filter_has_item_scope": true' in prompt
    assert '"scope_text": ""' in prompt
    assert "retrieval_plan" not in prompt
    assert "need_complete_sections" not in prompt
    assert "示例事项A" not in prompt


def test_knowledge_chain_ignores_removed_metadata_filter_keys():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(
        ChatRequest(
            query="身份证照片上传规范是什么",
            user_id=1,
            region_code="340000",
            biz_type="id_card_replace",
            metadata_filter={
                "source_type": "material_rule",
                "material_code": "ID_CARD_FRONT",
                "reject_code": "PHOTO_BLUR",
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


def test_rule_question_resolver_splits_multi_question():
    resolver = RuleQuestionResolver()

    plan = resolver.resolve(
        {
            "query": "办理居住证需要哪些材料？多久能办完，收费吗？",
            "normalized_query": "办理居住证需要哪些材料？多久能办完，收费吗？",
            "region_code": "340000",
        }
    )

    assert plan.is_multi_question is True
    assert [item["query"] for item in plan.questions] == ["办理居住证需要哪些材料", "多久能办完，收费吗"]


def test_rule_question_resolver_keeps_single_question_in_questions():
    resolver = RuleQuestionResolver()

    plan = resolver.resolve(
        {
            "query": "办理健康证需要哪些材料？",
            "normalized_query": "办理健康证需要哪些材料？",
            "region_code": "340000",
        }
    )

    assert plan.is_multi_question is False
    assert [item["query"] for item in plan.questions] == ["办理健康证需要哪些材料"]


def test_question_resolver_prompt_limits_over_splitting_and_markdown():
    assert "如果只有一个意图，同样返回 questions 数组" in QUESTION_RESOLVER_PROMPT
    assert "只能返回 JSON 对象" in QUESTION_RESOLVER_PROMPT
    assert "同一个业务问题中的限定条件" in QUESTION_RESOLVER_PROMPT
    assert "requires_ticket_context" in QUESTION_RESOLVER_PROMPT
    assert "ticket_fact_type" in QUESTION_RESOLVER_PROMPT
    assert "retrieval_plan" in QUESTION_RESOLVER_PROMPT
    assert "need_complete_sections" not in QUESTION_RESOLVER_PROMPT
    assert ("retrieval" "_focus") not in QUESTION_RESOLVER_PROMPT
    assert ("evidence" "_types") not in QUESTION_RESOLVER_PROMPT
    assert "retrieval_query" in QUESTION_RESOLVER_PROMPT
    assert "query 字段不负责对内容进行语义补全或改写" in QUESTION_RESOLVER_PROMPT
    assert "retrieval_query 才允许为了检索而补上下文" in QUESTION_RESOLVER_PROMPT
    assert "不负责对内容进行语义补全或改写" in QUESTION_RESOLVER_PROMPT
    assert "直接提取用户原始片段赋予 query 字段" in QUESTION_RESOLVER_PROMPT
    assert "补全原则" not in QUESTION_RESOLVER_PROMPT
    assert "示例事项甲" not in QUESTION_RESOLVER_PROMPT
    assert "示例证照乙" not in QUESTION_RESOLVER_PROMPT
    assert "示例事项A" in QUESTION_RESOLVER_PROMPT
    assert "草原" not in QUESTION_RESOLVER_PROMPT
    assert "施工" not in QUESTION_RESOLVER_PROMPT
    assert "防火办事流程" not in QUESTION_RESOLVER_PROMPT
    assert "健康证" not in QUESTION_RESOLVER_PROMPT


def test_llm_question_resolver_keeps_raw_query_when_model_expands():
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
                  "ticket_fact_type": "none",
                  "retrieval_plan": {"metadata_needs": ["item_scope"]},
                  "is_expanded": true,
                  "expansion_reason": "keyword_query_to_process_question",
                  "intent": "process_and_time",
                  "route": "ticket_context_unrelated"
                }
              ],
              "reason": "keyword_query_expanded"
            }
            """

    plan = LLMQuestionResolver(FakeLLM()).resolve({"query": "防火办事流程", "normalized_query": "防火办事流程"})

    assert plan.is_multi_question is False
    assert plan.questions[0]["original_query"] == "防火办事流程"
    assert plan.questions[0]["query"] == "防火办事流程"
    assert "need_complete_sections" not in plan.questions[0]["retrieval_plan"]
    assert plan.questions[0]["is_expanded"] is False
    assert plan.questions[0]["expansion_reason"] == ""
    assert "intent" not in plan.questions[0]
    assert "route" not in plan.questions[0]


def test_llm_question_resolver_allows_retrieval_query_context_without_changing_query():
    class FakeLLM:
        def invoke(self, prompt):
            self.prompt = prompt
            return """
            {
              "is_multi_question": true,
              "questions": [
                {
                  "original_query": "华侨回国定居审批多久能办完",
                  "query": "华侨回国定居审批多久能办完",
                  "retrieval_query": "华侨回国定居审批多久能办完",
                  "requires_ticket_context": false,
                  "ticket_fact_type": "none"
                },
                {
                  "original_query": "收费吗",
                  "query": "收费吗",
                  "retrieval_query": "华侨回国定居审批 收费吗",
                  "requires_ticket_context": false,
                  "ticket_fact_type": "none"
                }
              ],
              "reason": "short_followup_inherits_context_for_retrieval"
            }
            """

    llm = FakeLLM()
    plan = LLMQuestionResolver(llm).resolve(
        {"query": "华侨回国定居审批多久能办完？收费吗？", "normalized_query": "华侨回国定居审批多久能办完？收费吗？"}
    )

    assert plan.is_multi_question is True
    assert plan.questions[1]["query"] == "收费吗"
    assert plan.questions[1]["retrieval_query"] == "华侨回国定居审批 收费吗"
    assert "retrieval_query" in llm.prompt


def test_llm_question_resolver_keeps_ticket_related_query_unexpanded():
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

    plan = LLMQuestionResolver(FakeLLM()).resolve(
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
    assert plan.questions[0]["expansion_reason"] == ""


def test_single_question_workflow_keeps_resolver_raw_query_without_resolver_route():
    class ExpandingResolver:
        def resolve(self, state):
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

    workflow = DeclarationQAWorkflow(question_resolver=ExpandingResolver())
    response = workflow.invoke(ChatRequest(query="防火办事流程", user_id=1, region_code="340000"))

    assert response.multi_question is False
    assert response.question_resolution["questions"][0]["original_query"] == "防火办事流程"
    assert response.question_resolution["questions"][0]["query"] == "草原防火期内在草原上进行爆破、勘察和施工等活动审批的办理流程是什么"
    assert response.events[1]["data"]["normalized_query"] == "防火办事流程"
    assert response.question_resolution["questions"][0]["is_expanded"] is True
    assert response.route in {"ticket_context_unrelated", "clarification", "refuse"}
    assert response.route_decision.get("route") == response.route
    question_resolution_event = next(event for event in response.events if event["event"] == "question_resolution")
    assert question_resolution_event["data"]["expanded"] is True


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
                short_term_context="{}",
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
    assert "<SubQuery_1>" in audit["final_prompt"]
    assert "我的健康证办理到哪一步了" in audit["final_prompt"]
    assert "<SubQuery_2>" in audit["final_prompt"]
    assert "办理居住证需要哪些材料" in audit["final_prompt"]
    assert audit["multi_question"] is True
    assert len(audit["sub_tasks"]) == 2


def test_answer_prompt_uses_recent_session_context_only():
    workflow = DeclarationQAWorkflow()
    first = workflow.invoke(ChatRequest(query="取水许可是什么", user_id=1, region_code="340000"))

    second = workflow.invoke(
        ChatRequest(
            query="刚才我问的事项办理要多久",
            user_id=1,
            region_code="340000",
            session_id=first.session_id,
        )
    )
    audit = workflow.audit_store.get(second.audit_id)

    assert audit["short_term_context"]["metrics"]["recent_message_count"] == 2
    assert "短期上下文" in audit["final_prompt"]
    assert "取水许可是什么" in audit["final_prompt"]
    assert "刚才我问的事项办理要多久" in audit["final_prompt"]


def test_multi_question_uses_resolver_retrieval_query_for_short_followup():
    class FakeLLM:
        def invoke(self, prompt):
            return """
            {
              "is_multi_question": true,
              "questions": [
                {
                  "original_query": "华侨回国定居审批多久能办完",
                  "query": "华侨回国定居审批多久能办完",
                  "retrieval_query": "华侨回国定居审批多久能办完",
                  "requires_ticket_context": false,
                  "ticket_fact_type": "none"
                },
                {
                  "original_query": "收费吗",
                  "query": "收费吗",
                  "retrieval_query": "华侨回国定居审批 收费吗",
                  "requires_ticket_context": false,
                  "ticket_fact_type": "none"
                }
              ],
              "reason": "short_followup_inherits_context_for_retrieval"
            }
            """

    class CapturingRAGEngine(InMemoryRAGEngine):
        def __init__(self):
            super().__init__()
            self.queries = []

        def retrieve_many(self, queries, filters, *, retrieval_plan=None):
            self.queries.extend(query["text"] for query in queries)
            return super().retrieve_many(queries, filters, retrieval_plan=retrieval_plan)

        def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k=6, retrieval_plan=None):
            self.queries.extend(query["text"] for query in retrieval_queries)
            return super().retrieve_rewrites(
                original_query,
                retrieval_queries,
                filters,
                top_k=top_k,
                retrieval_plan=retrieval_plan,
            )

    engine = CapturingRAGEngine()
    engine.index_document(
        KnowledgeDocument(
            id="doc-overseas-chinese-settlement",
            title="华侨回国定居审批办事指南",
            content="华侨回国定居审批的法定办结时限为15个工作日，承诺办结时限为8个工作日。是否收费：否。",
            region_code="340000",
            policy_version="2025",
            metadata={"item_name": "华侨回国定居审批"},
        )
    )
    workflow = DeclarationQAWorkflow(
        rag_engine=engine,
        question_resolver=LLMQuestionResolver(FakeLLM()),
        max_sub_question_workers=1,
    )

    response = workflow.invoke(
        ChatRequest(
            query="华侨回国定居审批多久能办完？收费吗？",
            user_id=1,
            region_code="340000",
            policy_version="2025",
        )
    )

    assert response.multi_question is True
    assert response.question_resolution["questions"][1]["query"] == "收费吗"
    assert response.question_resolution["questions"][1]["retrieval_query"] == "华侨回国定居审批 收费吗"
    assert response.sub_tasks[1]["query"] == "收费吗"
    assert response.sub_tasks[1]["retrieval_query"] == "华侨回国定居审批 收费吗"
    assert any("华侨回国定居审批 收费吗" in query for query in engine.queries)


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


def test_question_resolver_drives_ticket_context_route_without_router():
    class TicketRelatedresolver:
        def resolve(self, state):
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
                    "reason": "resolver_ticket_related",
                    "provider": "test",
                    "payload": lambda self: {
                        "is_multi_question": self.is_multi_question,
                        "questions": self.questions,
                        "reason": self.reason,
                        "provider": self.provider,
                    },
                },
            )()

    workflow = DeclarationQAWorkflow(question_resolver=TicketRelatedresolver())

    response = workflow.invoke(
        ChatRequest(query="麻烦看一下这单现在什么情况", user_id=1, region_code="340000", apply_id="A1001")
    )

    assert response.route == "ticket_context_related"
    assert response.intent == "ticket_status"
    assert response.route_decision["entities"]["router_provider"] == "question_resolver"
    assert "部门审核中" in response.answer


def test_question_resolver_drives_knowledge_route_without_router():
    workflow = DeclarationQAWorkflow()

    response = workflow.invoke(ChatRequest(query="办理居住证需要哪些材料", user_id=1, region_code="340000"))

    assert response.route == "ticket_context_unrelated"
    assert response.intent == "material_requirements"
    assert response.route_decision["entities"]["router_provider"] == "question_resolver"


def test_bound_ticket_scopes_knowledge_retrieval_without_ticket_status_context():
    fire_context = {
        "apply_id": "AHZWFW-MOCK-FIRE-001",
        "user_id": 1,
        "biz_type": "grassland_fire_blasting_approval",
        "region_code": "340000",
        "status": "rejected",
        "current_node": "MATERIAL_REVIEW",
        "node_name": "材料预审",
        "payload": {
            "title": "防火办事流程",
            "official_item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批",
            "source_file": "item_30.md",
        },
    }

    class FireAdapter(InMemoryBusinessDataAdapter):
        def get_apply_context(self, *, user_id: int, apply_id: str):
            return dict(fire_context) if apply_id == fire_context["apply_id"] else None

        def verify_apply_access(self, *, user_id: int, apply_id: str) -> bool:
            return apply_id == fire_context["apply_id"]

    class CapturingRAGEngine(InMemoryRAGEngine):
        def __init__(self):
            super().__init__()
            self.queries = []

        def retrieve(self, query, filters, *, top_k=4, retrieval_plan=None):
            self.queries.append(query)
            return super().retrieve(query, filters, top_k=top_k, retrieval_plan=retrieval_plan)

        def retrieve_many(self, queries, filters, *, retrieval_plan=None):
            self.queries.extend(query["text"] for query in queries)
            return super().retrieve_many(queries, filters, retrieval_plan=retrieval_plan)

        def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k=6, retrieval_plan=None):
            self.queries.extend(query["text"] for query in retrieval_queries)
            return super().retrieve_rewrites(
                original_query,
                retrieval_queries,
                filters,
                top_k=top_k,
                retrieval_plan=retrieval_plan,
            )

    engine = CapturingRAGEngine()
    engine.index_document(
        KnowledgeDocument(
            id="doc-fire-time-limit",
            title="草原防火期内在草原上进行爆破、勘察和施工等活动审批办事指南",
            content=(
                "草原防火期内在草原上进行爆破、勘察和施工等活动审批的"
                "法定办结时限为20个工作日，承诺办结时限为10个工作日。"
            ),
            region_code="340000",
            policy_version="2025",
            metadata={"item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批"},
        )
    )

    class Knowledgeresolver:
        def resolve(self, state):
            return type(
                "Plan",
                (),
                {
                    "is_multi_question": False,
                    "questions": [
                        {
                            "id": "q1",
                            "original_query": "这个事情最快几天能拿结果？",
                            "query": "这个事情最快几天能拿结果？",
                            "requires_ticket_context": False,
                            "is_expanded": False,
                            "expansion_reason": "",
                        }
                    ],
                    "reason": "resolver_marks_time_limit_as_knowledge",
                    "provider": "test",
                    "payload": lambda self: {
                        "is_multi_question": self.is_multi_question,
                        "questions": self.questions,
                        "reason": self.reason,
                        "provider": self.provider,
                    },
                },
            )()

    workflow = DeclarationQAWorkflow(
        rag_engine=engine,
        business_adapter=FireAdapter(),
        question_resolver=Knowledgeresolver(),
    )

    response = workflow.invoke(
        ChatRequest(
            query="这个事情最快几天能拿结果？",
            user_id=1,
            region_code="340000",
            apply_id="AHZWFW-MOCK-FIRE-001",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.route == "ticket_context_unrelated"
    assert response.intent == "fee_time_limit"
    assert response.retrieval_chain == "knowledge_retrieval"
    assert response.business_snapshot == {}
    assert engine.queries
    assert any("这个事情最快几天能拿结果" in query for query in engine.queries)
    assert len(audit["retrieval_queries"]) == 3
    assert audit["retrieval_filter"] == {"item_name": "草原防火期内在草原上进行爆破、勘察和施工等活动审批"}
    assert all("草原防火期内在草原上进行爆破、勘察和施工等活动审批" not in item["text"] for item in audit["retrieval_queries"])
    assert audit["rag_query"] == "\n".join(item["text"] for item in audit["retrieval_queries"])
    assert "10个工作日" in response.answer


def test_multi_retrieve_uses_batch_retrieval_and_dedupes():
    class SlowRAGEngine(InMemoryRAGEngine):
        def __init__(self):
            super().__init__()
            self.batch_queries = []

        def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k=6, retrieval_plan=None):
            time.sleep(0.12)
            self.batch_queries = [query["text"] for query in retrieval_queries]
            return RetrievalResult(
                sources=[],
                filter=filters,
                status={
                    "query": original_query,
                    "query_count": len(retrieval_queries),
                    "queries": [
                        {
                            "id": query.get("id") or "",
                            "strategy": query.get("strategy") or "",
                            "query": query.get("text") or "",
                            "top_k": int(query.get("top_k") or 3),
                            "source_count": 0,
                            "channels": {"dense": 0, "bm25": 0},
                        }
                        for query in retrieval_queries
                    ],
                    "source_count": 0,
                    "retrieval_mode": "batch_multi_query_rewrite_fused",
                    "batch_mode": True,
                },
            )

    workflow = DeclarationQAWorkflow(rag_engine=SlowRAGEngine(), max_retrieval_workers=3)
    state = {
        "retrieval_queries": [
            {"id": "rq1", "text": "办理时限", "strategy": "a", "top_k": 3},
            {"id": "rq2", "text": "办理时限", "strategy": "duplicate", "top_k": 3},
            {"id": "rq3", "text": "承诺办结时限", "strategy": "b", "top_k": 3},
            {"id": "rq4", "text": "法定办结时限", "strategy": "c", "top_k": 3},
        ],
    }

    started = time.perf_counter()
    result = workflow.multi_retrieve(state, {"item_code": "ITEM-001"})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.28
    assert result.status["retrieval_mode"] == "batch_multi_query_rewrite_fused"
    assert result.status["original_query_count"] == 4
    assert result.status["deduped_query_count"] == 3
    assert result.status["retrieval_parallelism"] == 1
    assert result.status["batch_mode"] is True
    assert [item["query"] for item in result.status["queries"]] == ["办理时限", "承诺办结时限", "法定办结时限"]


def test_image_only_query_skips_text_retrieval_and_rewrite():
    class FailIfCalledRewriter:
        name = "fail"

        def rewrite(self, state):
            raise AssertionError("image-only query should not use text query rewrite")

    class ImageOnlyRAGEngine(InMemoryRAGEngine):
        def __init__(self):
            super().__init__()
            self.image_calls = []

        def retrieve_image(self, image_path, text, filters, *, top_k=4, retrieval_plan=None):
            self.image_calls.append((image_path, text, filters, top_k, retrieval_plan))
            return RetrievalResult(sources=[], filter=filters, status={"cache": {"hit": False}, "channels": {"dense": 0}})

        def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k=6, retrieval_plan=None):
            raise AssertionError("image-only query should not use text rewrite retrieval")

        def retrieve_many(self, queries, filters, *, retrieval_plan=None):
            raise AssertionError("image-only query should not use text batch retrieval")

        def retrieve(self, query, filters, *, top_k=4, retrieval_plan=None):
            raise AssertionError("image-only query should not use text fallback retrieval")

    engine = ImageOnlyRAGEngine()
    workflow = DeclarationQAWorkflow(rag_engine=engine, query_rewriter=FailIfCalledRewriter())
    state = {
        "query": "请根据这张图片检索相关申报材料",
        "normalized_query": "请根据这张图片检索相关申报材料",
        "query_image": {"file_path": "C:/tmp/material.png", "text_present": False},
        "retrieval_queries": [{"id": "rq1", "text": "请根据这张图片检索相关申报材料", "strategy": "fallback", "top_k": 4}],
        "retrieval_plan": {},
        "route": "ticket_context_unrelated",
    }

    rewritten_state = workflow.query_rewrite(state)
    result = workflow.multi_retrieve(rewritten_state, {"policy_version": "2025"})

    assert rewritten_state["retrieval_queries"] == []
    assert rewritten_state["query_rewrite"]["provider"] == "image_only"
    assert rewritten_state["query_rewrite"]["reason"] == "image_only_skip_text_rewrite"
    assert engine.image_calls
    assert engine.image_calls[0][0] == "C:/tmp/material.png"
    assert engine.image_calls[0][1] == ""
    assert result.status["retrieval_mode"] == "image_only"
    assert result.status["query_modalities"] == ["image"]
    assert result.status["query_count"] == 0
    assert result.status["text_retrieval_skipped"] is True
    assert result.status["queries"][0]["strategy"] == "image_query"


def test_image_only_graph_skips_text_resolution_and_rewrite():
    class FailIfCalledResolver:
        name = "fail"

        def resolve(self, state):
            raise AssertionError("image-only query should not use question resolution")

    class FailIfCalledRewriter:
        name = "fail"

        def rewrite(self, state):
            raise AssertionError("image-only query should not use text query rewrite")

    class ImageOnlyRAGEngine(InMemoryRAGEngine):
        def __init__(self):
            super().__init__()
            self.image_calls = []

        def retrieve_image(self, image_path, text, filters, *, top_k=4, retrieval_plan=None):
            self.image_calls.append((image_path, text, filters, top_k, retrieval_plan))
            return RetrievalResult(
                sources=[],
                filter=filters,
                status={
                    "query": "",
                    "retrieval_mode": "image_only",
                    "cache": {"hit": False},
                    "channels": {"dense": 0, "bm25": 0, "rerank": 0},
                    "queries": [{"id": "image", "strategy": "image_query", "query": ""}],
                    "text_retrieval_skipped": True,
                },
            )

        def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k=6, retrieval_plan=None):
            raise AssertionError("image-only query should not use text rewrite retrieval")

        def retrieve_many(self, queries, filters, *, retrieval_plan=None):
            raise AssertionError("image-only query should not use text batch retrieval")

        def retrieve(self, query, filters, *, top_k=4, retrieval_plan=None):
            raise AssertionError("image-only query should not use text fallback retrieval")

    engine = ImageOnlyRAGEngine()
    workflow = DeclarationQAWorkflow(
        rag_engine=engine,
        question_resolver=FailIfCalledResolver(),
        query_rewriter=FailIfCalledRewriter(),
    )

    response = workflow.invoke(
        ChatRequest(
            query="",
            query_image=ChatImagePayload(file_path="C:/tmp/material.png", text_present=False),
        )
    )

    assert engine.image_calls
    assert engine.image_calls[0][1] == ""
    assert response.question_resolution == {}
    assert response.rag_status["retrieval_mode"] == "image_only"
    assert response.rag_status["text_retrieval_skipped"] is True
    assert any(event["event"] == "image_only_prepare" for event in response.events)
    assert not any(event["event"] == "question_resolution" for event in response.events)
    assert not any(event["event"] == "query_rewrite" for event in response.events)


def test_multi_retrieve_falls_back_to_parallel_when_batch_fails():
    class SlowRAGEngine(InMemoryRAGEngine):
        def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k=6, retrieval_plan=None):
            raise RuntimeError("fused unavailable")

        def retrieve_many(self, queries, filters, *, retrieval_plan=None):
            raise RuntimeError("batch unavailable")

        def retrieve(self, query, filters, *, top_k=4, retrieval_plan=None):
            time.sleep(0.12)
            return RetrievalResult(sources=[], filter=filters, status={"cache": {"hit": False}, "channels": {}})

    workflow = DeclarationQAWorkflow(rag_engine=SlowRAGEngine(), max_retrieval_workers=3)
    state = {
        "retrieval_queries": [
            {"id": "rq1", "text": "办理时限", "strategy": "a", "top_k": 3},
            {"id": "rq2", "text": "承诺办结时限", "strategy": "b", "top_k": 3},
            {"id": "rq3", "text": "法定办结时限", "strategy": "c", "top_k": 3},
        ],
    }

    started = time.perf_counter()
    result = workflow.multi_retrieve(state, {"item_code": "ITEM-001"})
    elapsed = time.perf_counter() - started

    assert elapsed < 0.28
    assert result.status["retrieval_mode"] == "parallel_multi_query_rewrite"
    assert result.status["batch_degraded"] == "RuntimeError"


def test_rag_retrieve_many_batches_embedding_and_vector_search():
    class BatchEmbedding:
        name = "batch-test"
        dimension = 96

        def __init__(self):
            self.document_batches = []

        def embed_query(self, text):
            return [1.0] + [0.0] * 95

        def embed_documents(self, texts):
            self.document_batches.append(list(texts))
            return [[1.0] + [0.0] * 95 for _ in texts]

    class BatchVectorIndex:
        backend = "batch-test"

        def __init__(self):
            self.batch_calls = []

        def upsert(self, chunk, vector):
            pass

        def delete_by_doc(self, doc_id):
            pass

        def search(self, vector, filters, *, limit):
            raise AssertionError("single search should not be used")

        def search_batch(self, vectors, filters, *, limit):
            self.batch_calls.append((len(vectors), filters, limit))
            return [[("chunk-batch", 0.99)] for _ in vectors]

        def clear(self):
            pass

    embedding = BatchEmbedding()
    vector_index = BatchVectorIndex()
    engine = InMemoryRAGEngine(embedding_provider=embedding, vector_index=vector_index)
    engine.load_indexed_data(
        [
            KnowledgeDocument(
                id="doc-batch",
                title="批量检索测试指南",
                content="办理时限为10个工作日，承诺办结时限为5个工作日。",
            )
        ],
        [
            KnowledgeChunk(
                id="chunk-batch",
                doc_id="doc-batch",
                parent_id="",
                chunk_type="child",
                title="批量检索测试指南",
                content="办理时限为10个工作日，承诺办结时限为5个工作日。",
            )
        ],
    )

    results = engine.retrieve_many(
        [
            {"text": "办理时限", "top_k": 2},
            {"text": "承诺办结时限", "top_k": 2},
        ],
        {},
    )

    assert embedding.document_batches == [["办理时限", "承诺办结时限"]]
    assert vector_index.batch_calls == [(2, {"doc_status": "published"}, 8)]
    assert len(results) == 2
    assert all(result.status["batch_mode"] is True for result in results)


def test_select_answer_sources_preserves_complete_process_steps():
    section_id = "doc-flow#办理流程"

    def source(
        chunk_id: str,
        node_title: str,
        metadata_chunk_type: str,
        node_order: int,
        *,
        score: float,
        rerank: float,
    ) -> RetrievedSource:
        return RetrievedSource(
            chunk=KnowledgeChunk(
                id=chunk_id,
                doc_id="doc-flow",
                parent_id="",
                chunk_type="child",
                title="取水许可办事指南",
                content=f"{node_title}内容",
                metadata={
                    "chunk_type": metadata_chunk_type,
                    "section_title": "办理流程" if metadata_chunk_type != "material_item" else "申请材料",
                    "parent_section_id": section_id if metadata_chunk_type != "material_item" else "doc-flow#申请材料",
                    "node_title": node_title,
                    "node_order": node_order,
                    "indexable": True,
                },
            ),
            score=score,
            rerank_score=rerank,
            retrieval_channel="test",
        )

    sources = [
        source("flow-summary", "办理流程说明", "section_leaf", 9, score=0.97, rerank=0.97),
        source("material-1", "取水许可申请书", "material_item", 1, score=0.92, rerank=0.92),
        source("material-2", "营业执照或身份证", "material_item", 2, score=0.91, rerank=0.91),
        source("step-1", "受理", "process_step", 10, score=0.97, rerank=0.0),
        source("step-2", "审查", "process_step", 11, score=0.97, rerank=0.0),
        source("step-3", "决定", "process_step", 12, score=0.97, rerank=0.0),
        source("step-4", "办结", "process_step", 13, score=0.97, rerank=0.0),
        source("step-5", "送达", "process_step", 14, score=0.97, rerank=0.0),
    ]

    selected = select_answer_sources({item.chunk.id: item for item in sources}, "取水许可办理流程", limit=6)

    assert [item.chunk.metadata["node_title"] for item in selected] == ["办理流程说明", "受理", "审查", "决定", "办结", "送达"]


def test_collect_sub_question_uses_compiled_sub_question_graph():
    class FakeSubQuestionGraph:
        def __init__(self):
            self.invocations = []

        def invoke(self, state):
            self.invocations.append(dict(state))
            return {
                **state,
                "route": "ticket_context_unrelated",
                "intent": "knowledge_qa",
                "sub_intent": "knowledge_qa",
                "retrieval_chain": "knowledge_retrieval",
                "evidence_ok": True,
                "refused": False,
                "rag_query": state.get("normalized_query") or "",
            }

    workflow = DeclarationQAWorkflow()
    fake_graph = FakeSubQuestionGraph()
    workflow.sub_question_graph = fake_graph
    question = {"id": "q1", "query": "问题一", "requires_ticket_context": False}
    state = workflow.state_for_sub_question({"events": [], "request_biz_type": None}, question)

    task = workflow.collect_sub_question(state, question, 1)

    assert len(fake_graph.invocations) == 1
    assert fake_graph.invocations[0]["query"] == "问题一"
    assert task["id"] == "q1"
    assert task["route"] == "ticket_context_unrelated"
    assert task["status"] == "ready"


def test_multi_question_collect_runs_sub_questions_in_parallel():
    class SlowWorkflow(DeclarationQAWorkflow):
        def state_for_sub_question(self, state, question):
            return {**state, "query": question["query"], "normalized_query": question["query"]}

        def collect_sub_question(self, state, question, index):
            time.sleep(0.12)
            return {
                "id": question.get("id") or f"q{index}",
                "query": question["query"],
                "route": "ticket_context_unrelated",
                "intent": "knowledge_qa",
                "status": "ready",
                "sources": [],
                "rag_query": question["query"],
            }

    workflow = SlowWorkflow(max_sub_question_workers=2)
    state = {
        "sub_tasks": [
            {"id": "q1", "query": "问题一", "requires_ticket_context": False},
            {"id": "q2", "query": "问题二", "requires_ticket_context": False},
        ],
        "events": [],
    }

    started = time.perf_counter()
    result = workflow.multi_question_collect(state)
    elapsed = time.perf_counter() - started

    event = next(item for item in result["events"] if item["event"] == "multi_question_collected")
    assert elapsed < 0.22
    assert [task["id"] for task in result["sub_tasks"]] == ["q1", "q2"]
    assert event["data"]["collection_mode"] == "parallel"
    assert event["data"]["sub_question_runner"] == "langgraph"
    assert event["data"]["sub_question_parallelism"] == 2


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
                "biz_type": "residence_permit",
                "material_code": "RESIDENCE_PROOF",
                "reject_code": "MISSING_RESIDENCE_PROOF",
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
        "declaration_qa.assets",
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


def test_knowledge_chain_uses_full_corpus_without_ticket_context():
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


def test_selected_ticket_scopes_general_question_with_item_metadata_filter():
    adapter = InMemoryBusinessDataAdapter()
    adapter._applications["A1001"]["payload"] = {
        "official_item_name": "示例事项A",
        "item_code": "ITEM001",
    }
    engine = InMemoryRAGEngine()
    engine.index_document(
        KnowledgeDocument(
            id="doc-example-item",
            title="示例事项A办事指南",
            content="示例事项A的承诺办结时限为3个工作日。",
            metadata={"item_code": "ITEM001", "item_name": "示例事项A"},
        )
    )
    workflow = DeclarationQAWorkflow(rag_engine=engine, business_adapter=adapter)

    response = workflow.invoke(
        ChatRequest(
            query="这个事情最快几天能拿结果",
            user_id=1,
            region_code="340000",
            apply_id="A1001",
        )
    )
    audit = workflow.audit_store.get(response.audit_id)

    assert response.route == "ticket_context_related"
    assert audit["retrieval_filter"] == {"item_code": "ITEM001"}
    assert audit["rag_status"]["metadata_filtering"] is True
    assert all("示例事项A" not in query["query"] for query in audit["rag_status"]["queries"])
    assert "3个工作日" in response.answer


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
    store = BusinessRepository(FakeBusinessEngine(rows))

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
