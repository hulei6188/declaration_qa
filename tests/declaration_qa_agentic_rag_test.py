from __future__ import annotations

from declaration_qa.api.schemas import ChatRequest
from declaration_qa.application.tools import (
    KnowledgeSearchTool,
    TicketDetailQueryTool,
    TicketListQueryTool,
    ToolExecutionContext,
    ToolRegistry,
)
from declaration_qa.application.agent_planner import AgentDecision, choose_application_id
from declaration_qa.application.workflow import DeclarationQAWorkflow
from declaration_qa.application.workflow_agent_nodes import agent_tool_call_allowed, agent_tool_call_denial_reason
from declaration_qa.core.business import InMemoryBusinessDataAdapter
from declaration_qa.core.domain import KnowledgeChunk, RetrievalResult, RetrievedSource


class CapturingKnowledgeRag:
    rerank_score_threshold = 0.7
    rerank_max_sources = 36

    def __init__(self):
        self.calls = []
        chunk = KnowledgeChunk(
            id="chunk-materials",
            doc_id="doc-materials",
            parent_id="",
            chunk_type="child",
            title="健康证办理指南",
            content="申请健康证需要身份证明和体检材料。",
            region_code="340000",
            policy_version="2025",
        )
        self.result = RetrievalResult(
            sources=[RetrievedSource(chunk=chunk, score=0.9, rerank_score=0.91, retrieval_channel="hybrid")],
            filter={},
            status={"retrieval_mode": "test", "score_cutoff": {"threshold": 0.7}},
        )

    def retrieve_rewrites(self, original_query, retrieval_queries, filters, *, top_k, retrieval_plan):
        self.calls.append(
            {
                "query": original_query,
                "queries": retrieval_queries,
                "filters": dict(filters),
                "retrieval_plan": dict(retrieval_plan),
            }
        )
        return RetrievalResult(self.result.sources, dict(filters), self.result.status)

    def retrieve(self, query, filters, *, top_k, retrieval_plan):
        self.calls.append(
            {"query": query, "queries": [], "filters": dict(filters), "retrieval_plan": dict(retrieval_plan)}
        )
        return RetrievalResult(self.result.sources, dict(filters), self.result.status)


def test_tool_registry_exposes_item_scoped_ticket_tools_without_user_id_argument():
    adapter = InMemoryBusinessDataAdapter()
    registry = ToolRegistry()
    registry.register(TicketListQueryTool(adapter))
    registry.register(TicketDetailQueryTool(adapter))

    definitions = {item["function"]["name"]: item["function"] for item in registry.definitions()}

    assert set(definitions) == {"ticket_list_query", "ticket_detail_query"}
    assert "user_id" not in definitions["ticket_list_query"]["parameters"].get("properties", {})
    assert "region_code" not in definitions["ticket_list_query"]["parameters"].get("properties", {})
    assert "user_id" not in definitions["ticket_detail_query"]["parameters"].get("properties", {})
    assert "ticket_id" not in definitions["ticket_detail_query"]["parameters"].get("properties", {})

    context = ToolExecutionContext(user_id=1, region_code="340000")
    listing = registry.execute("ticket_list_query", {}, context, call_id="list-1")
    detail = registry.execute(
        "ticket_detail_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "detail_type": "rejection"},
        context,
        call_id="detail-1",
    )

    assert listing.ok is True
    assert any(item["item_code"] == "ITEM-RESIDENCE-PERMIT" for item in listing.data["items"])
    assert detail.ok is True
    assert detail.data["applications"][0]["application_id"] == 2002
    assert len(detail.data["applications"][0]["status_history"]) == 2
    assert detail.data["applications"][0]["matching_status_events"][0]["reason"] == "缺少有效居住证明材料"


def test_knowledge_tool_schema_requires_structured_agent_queries():
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(lambda request, context: {"sources": [], "executed_queries": []}))

    parameters = registry.definitions()[0]["function"]["parameters"]

    assert "queries" in parameters["properties"]
    assert "query_focus" not in parameters["properties"]
    assert "user_id" not in parameters["properties"]
    assert "top_k" not in parameters["properties"]


def test_chat_request_has_no_bound_ticket_fields():
    fields = ChatRequest.model_json_schema().get("properties", {})

    assert "apply_id" not in fields
    assert "apply_context" not in fields


def test_ticket_detail_returns_every_application_and_every_status_change_for_item():
    adapter = InMemoryBusinessDataAdapter()
    previous = dict(adapter._applications["A2002"])
    previous.update(
        {
            "application_id": 1900,
            "status": "completed",
            "current_node": "FINISH",
            "node_name": "办结",
            "submitted_at": "2025-03-02T08:40:00+08:00",
            "completed_at": "2025-03-20T16:10:00+08:00",
            "updated_at": "2025-03-20T16:10:00+08:00",
            "status_history": [
                {"event_id": 19001, "sequence_no": 1, "from_status": None, "status": "submitted", "changed_at": "2025-03-02T08:40:00+08:00"},
                {"event_id": 19002, "sequence_no": 2, "from_status": "submitted", "status": "accepted", "changed_at": "2025-03-03T09:20:00+08:00"},
                {"event_id": 19003, "sequence_no": 3, "from_status": "accepted", "status": "completed", "changed_at": "2025-03-20T16:10:00+08:00"},
            ],
        }
    )
    adapter._applications["A1900"] = previous

    applications = adapter.get_applications_by_item(user_id=1, item_code="ITEM-RESIDENCE-PERMIT")

    assert {item["application_id"] for item in applications} == {1900, 2002}
    assert [event["sequence_no"] for event in next(item for item in applications if item["application_id"] == 1900)["status_history"]] == [1, 2, 3]
    assert choose_application_id("看一下2025年那次申请", applications) == 1900
    assert choose_application_id("看一下上一次申请", applications) == 1900


def test_ticket_detail_is_scoped_to_trusted_user_context():
    registry = ToolRegistry()
    registry.register(TicketDetailQueryTool(InMemoryBusinessDataAdapter()))

    result = registry.execute(
        "ticket_detail_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT"},
        ToolExecutionContext(user_id=999, region_code="340000"),
    )

    assert result.ok is False
    assert result.status == "not_found"
    assert result.data["applications"] == []


def test_tool_policy_enforces_business_and_knowledge_isolation():
    assert agent_tool_call_allowed("ticket_list_query", "健康证需要哪些材料") is False
    assert agent_tool_call_allowed("knowledge_search", "我的居住证申请进度") is True
    assert agent_tool_call_allowed("knowledge_search", "我的居住证申请被退回，怎么补正") is True
    assert "requires a successful" in agent_tool_call_denial_reason(
        "ticket_detail_query",
        "我的居住证申请进度",
        arguments={"item_code": "ITEM-RESIDENCE-PERMIT"},
        prior_results=[],
    )
    assert "was not returned" in agent_tool_call_denial_reason(
        "ticket_detail_query",
        "我的居住证申请进度",
        arguments={"item_code": "ITEM-OTHER"},
        prior_results=[
            {
                "tool_name": "ticket_list_query",
                "ok": True,
                "data": {"items": [{"item_code": "ITEM-RESIDENCE-PERMIT"}]},
            }
        ],
    )
    assert "before observing business" in agent_tool_call_denial_reason(
        "knowledge_search",
        "我的居住证申请被退回，怎么补正",
        arguments={"queries": [{"text": "居住证补正材料", "strategy": "hybrid"}]},
        prior_results=[{"tool_name": "ticket_list_query", "ok": True, "status": "success", "data": {}}],
    )


def test_agent_ticket_question_selects_item_and_application_without_bound_ticket():
    workflow = DeclarationQAWorkflow(rag_engine=CapturingKnowledgeRag())

    response = workflow.invoke(
        ChatRequest(
            query="我的居住证申请为什么被退回",
            user_id=1,
            region_code="340000",
        )
    )

    tool_names = [item["tool_name"] for item in response.tool_results]
    assert tool_names == ["ticket_list_query", "ticket_detail_query"]
    assert response.route == "agentic_rag"
    assert response.intent == "rejection_reason"
    assert response.evidence_decision["business_tool_ok"] is True
    assert response.business_snapshot["item_code"] == "ITEM-RESIDENCE-PERMIT"
    assert response.business_snapshot["selected_application_id"] == 2002
    assert "缺少有效居住证明材料" in response.answer


def test_agent_can_answer_application_list_without_forcing_a_detail_selection():
    workflow = DeclarationQAWorkflow(rag_engine=CapturingKnowledgeRag())

    response = workflow.invoke(ChatRequest(query="我有哪些历史申请", user_id=1, region_code="340000"))

    assert response.intent == "ticket_list"
    assert [item["tool_name"] for item in response.tool_results] == ["ticket_list_query"]
    assert response.evidence_decision["business_tool_ok"] is True
    assert response.business_snapshot == {}
    assert "健康证办理" in response.answer
    assert "居住证办理" in response.answer


def test_agent_knowledge_question_uses_only_knowledge_tool_and_frozen_query():
    rag = CapturingKnowledgeRag()
    workflow = DeclarationQAWorkflow(rag_engine=rag)

    response = workflow.invoke(
        ChatRequest(
            query="健康证需要哪些材料",
            user_id=1,
            region_code="340000",
            metadata_filter={"region_code": "340000"},
        )
    )

    assert [item["tool_name"] for item in response.tool_results] == ["knowledge_search"]
    assert response.sources[0].chunk_id == "chunk-materials"
    assert rag.calls
    assert rag.calls[0]["query"] == "健康证需要哪些材料"
    assert rag.calls[0]["filters"] == {"region_code": "340000"}
    assert response.agent_query_plan["executed_queries"][0]["strategy"] == "hybrid"
    assert "ticket_fact_type" not in rag.calls[0]["retrieval_plan"]
    assert "A2002" not in str(rag.calls[0])


def test_mixed_ticket_and_guidance_question_keeps_business_result_out_of_rag_query():
    rag = CapturingKnowledgeRag()
    workflow = DeclarationQAWorkflow(rag_engine=rag)

    response = workflow.invoke(
        ChatRequest(
            query="我的居住证申请为什么被退回，应该怎么补正",
            user_id=1,
            region_code="340000",
        )
    )

    assert [item["tool_name"] for item in response.tool_results] == [
        "ticket_list_query",
        "knowledge_search",
        "ticket_detail_query",
    ]
    assert response.evidence_decision["business_tool_ok"] is True
    assert response.evidence_decision["knowledge_ok"] is True
    assert rag.calls[0]["query"] == "我的居住证申请为什么被退回，应该怎么补正"
    serialized_call = str(rag.calls[0])
    assert "缺少有效居住证明材料" not in serialized_call
    assert "MISSING_RESIDENCE_PROOF" not in serialized_call


def test_agent_final_response_is_used_without_separate_answer_generation():
    class FinalAnswerPlanner:
        def decide(self, state, tool_definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [
                        {
                            "id": "knowledge",
                            "name": "knowledge_search",
                            "arguments": {
                                "queries": [
                                    {"text": "健康证申请材料", "strategy": "hybrid", "aspect": "materials"}
                                ]
                            },
                        }
                    ],
                    intent="material_requirements",
                )
            return AgentDecision(
                "finish",
                intent="material_requirements",
                provider="fake_agent",
                final_answer="这是 Agent 根据知识工具证据生成的最终答案。",
                final_prompt="agent prompt",
            )

    workflow = DeclarationQAWorkflow(rag_engine=CapturingKnowledgeRag(), agent_planner=FinalAnswerPlanner())

    response = workflow.invoke(ChatRequest(query="健康证需要哪些材料", user_id=1))

    assert response.answer == "这是 Agent 根据知识工具证据生成的最终答案。"
    assert response.agent_query_plan["requested_queries"][0]["aspect"] == "materials"
