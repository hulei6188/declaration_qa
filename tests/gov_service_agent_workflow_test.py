from pathlib import Path

import pytest
from pydantic import ValidationError

from gov_service_agent.api.schemas import ChatRequest
from gov_service_agent.application.agent import AgentDecision
from gov_service_agent.application.workflow import GovServiceAgentWorkflow
from gov_service_agent.core.business import InMemoryBusinessDataAdapter
from gov_service_agent.rag.engine.seed import build_seed_rag_engine
from gov_service_agent.rag.router import parse_json_object


class TicketScriptAgent:
    def __init__(self, item_code, detail_type="status"):
        self.item_code = item_code
        self.detail_type = detail_type

    def decide(self, state, _tool_definitions):
        completed = {item["tool_name"] for item in state.get("tool_results") or [] if item.get("ok")}
        if "ticket_list_query" not in completed:
            return AgentDecision(
                "tools",
                [{"id": "list", "name": "ticket_list_query", "arguments": {}}],
            )
        if "ticket_detail_query" not in completed:
            return AgentDecision(
                "tools",
                [
                    {
                        "id": "detail",
                        "name": "ticket_detail_query",
                        "arguments": {
                            "item_codes": [self.item_code],
                            "detail_type": self.detail_type,
                        },
                    }
                ],
            )
        return AgentDecision("finish", final_answer="该申请当前状态和完整历史已查询，请查看申请详情。[业务2]")


def test_chat_routes_have_no_ticket_selection_endpoint():
    routes = Path("src/declaration_qa/api/server/routes/chat.py").read_text(encoding="utf-8")

    assert '"/api/declaration-qa/chat"' in routes
    assert '"/api/declaration-qa/chat/stream"' in routes
    assert '"/api/declaration-qa/chat/confirm/stream"' in routes
    assert "/tickets" not in routes


def test_chat_request_rejects_legacy_bound_ticket_fields():
    with pytest.raises(ValidationError):
        ChatRequest(query="我的申请进度", apply_id="A1001")
    with pytest.raises(ValidationError):
        ChatRequest(query="我的申请进度", apply_context={"apply_id": "A1001"})


def test_agentic_workflow_emits_tool_decisions_without_manual_selection_fields():
    workflow = GovServiceAgentWorkflow(
        rag_engine=build_seed_rag_engine(),
        business_adapter=InMemoryBusinessDataAdapter(),
        agent=TicketScriptAgent("ITEM-RESIDENCE-PERMIT", "rejection"),
    )

    response = workflow.invoke(ChatRequest(query="我的居住证申请为什么被退回", user_id=1))

    assert response.route == "agentic_rag"
    assert [result["tool_name"] for result in response.tool_results] == [
        "ticket_list_query",
        "ticket_detail_query",
    ]
    assert response.business_snapshot["selected_application_id"] == 2002
    assert "申请详情" in response.answer
    audit = workflow.audit_store.get(response.audit_id)
    business_registry = audit["evidence_registry"]["business"]
    assert business_registry[0]["evidence_id"] == "业务1"
    assert business_registry[0]["call_id"] == "list"
    assert business_registry[0]["tool_name"] == "ticket_list_query"
    assert "ITEM-RESIDENCE-PERMIT" in business_registry[0]["business_key"]["item_code"]
    assert business_registry[1] == {
        "evidence_id": "业务2",
        "call_id": "detail",
        "tool_name": "ticket_detail_query",
        "business_key": {"application_id": [2002]},
    }
    serialized = response.model_dump()
    assert "need_ticket_selection" not in serialized
    assert "tickets" not in serialized
    assert "ticket_context" not in serialized


def test_stream_done_event_contains_agent_result_not_ticket_picker():
    workflow = GovServiceAgentWorkflow(
        rag_engine=build_seed_rag_engine(),
        business_adapter=InMemoryBusinessDataAdapter(),
        agent=TicketScriptAgent("ITEM-HEALTH-CERTIFICATE"),
    )

    events = list(workflow.stream(ChatRequest(query="我的健康证申请进度", user_id=1)))
    done = next(event["data"] for event in events if event["event"] == "done")

    assert [event["event"] for event in events].count("evidence") == 0
    assert done["agent_iteration"] >= 2
    assert done["selected_item_code"] == "ITEM-HEALTH-CERTIFICATE"
    assert done["selected_application_id"] == 1001
    assert "need_ticket_selection" not in done
    assert "tickets" not in done


def test_ticket_list_only_finishes_without_a_gate():
    class TicketListOnlyAgent:
        def decide(self, state, _tool_definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [{"id": "list", "name": "ticket_list_query", "arguments": {}}],
                )
            return AgentDecision("finish", final_answer="已查询到你的申请事项。[业务1]")

    workflow = GovServiceAgentWorkflow(
        rag_engine=build_seed_rag_engine(),
        business_adapter=InMemoryBusinessDataAdapter(),
        agent=TicketListOnlyAgent(),
    )

    events = list(workflow.stream(ChatRequest(query="我现在有哪些工单", user_id=1)))

    assert [event["event"] for event in events].count("evidence") == 0
    assert [event["event"] for event in events].count("agent_decision") == 2


def test_agentic_schema_and_migration_define_complete_application_history():
    schema = Path("sql/postgresql/declaration_qa_postgresql_all.sql").read_text(encoding="utf-8")

    for token in ["item_code", "sequence_no", "from_status", "occurred_at", "source_event_id"]:
        assert token in schema
    for token in ["agent_plan", "agent_query_plan", "agent_answer_provider", "tool_results", "agent_selected_application_id"]:
        assert token in schema


def test_parse_json_object_accepts_fenced_model_output():
    assert parse_json_object('```json\n{"route":"agentic_rag"}\n```') == {"route": "agentic_rag"}
