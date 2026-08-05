from gov_service_agent.api.schemas import ChatRequest
from gov_service_agent.application.routing import StaticIntentRouter
from gov_service_agent.application.prompts import build_agent_system_prompt
from gov_service_agent.application.state_machine import transition_dialog_task
from gov_service_agent.application.tool_policy import (
    allowed_tool_names,
    filter_tool_definitions_for_route,
    partition_tool_calls,
)
from gov_service_agent.application.workflow import GovServiceAgentWorkflow


def route(**overrides):
    decision = {
        "domain": "knowledge",
        "operation": "consult",
        "confidence": 1.0,
        "reason": "test",
    }
    decision.update(overrides)
    return StaticIntentRouter(decision)


def test_dynamic_tool_policy_exposes_only_the_routed_domain():
    definitions = [
        {"type": "function", "function": {"name": "knowledge_search"}},
        {"type": "function", "function": {"name": "application_submit"}},
        {"type": "function", "function": {"name": "appointment_hall_query"}},
        {"type": "function", "function": {"name": "appointment_book"}},
    ]
    application = {"domain": "application", "operation": "start"}
    appointment = {"domain": "appointment", "operation": "book"}

    application_names = {
        item["function"]["name"] for item in filter_tool_definitions_for_route(definitions, application)
    }
    appointment_names = {
        item["function"]["name"] for item in filter_tool_definitions_for_route(definitions, appointment)
    }

    assert application_names == {"knowledge_search", "application_submit"}
    assert appointment_names == {"knowledge_search", "appointment_hall_query", "appointment_book"}
    accepted, denied = partition_tool_calls(
        [{"name": "application_submit"}, {"name": "appointment_book"}],
        application,
    )
    assert [item["name"] for item in accepted] == ["application_submit"]
    assert [item["name"] for item in denied] == ["appointment_book"]
    accepted, denied = partition_tool_calls(
        [{"name": "appointment_cancel"}, {"name": "appointment_book"}],
        {"domain": "appointment", "operation": "cancel"},
    )
    assert [item["name"] for item in accepted] == ["appointment_cancel"]
    assert [item["name"] for item in denied] == ["appointment_book"]
    assert allowed_tool_names({"domain": "appointment", "operation": "book"}) == {
        "knowledge_search",
        "appointment_hall_query",
        "appointment_slot_query",
        "appointment_book",
    }
    assert allowed_tool_names({"domain": "appointment", "operation": "cancel"}) == {
        "appointment_list_query",
        "appointment_cancel",
    }


def test_dynamic_prompt_contains_only_the_current_domain_rules():
    knowledge = build_agent_system_prompt(
        {"route_decision": {"domain": "knowledge", "operation": "consult"}}
    )
    application = build_agent_system_prompt(
        {"route_decision": {"domain": "application", "operation": "start"}}
    )
    booking = build_agent_system_prompt(
        {"route_decision": {"domain": "appointment", "operation": "book"}}
    )
    conversation = build_agent_system_prompt(
        {"route_decision": {"domain": "conversation", "operation": "chat"}}
    )

    assert "knowledge_search" in knowledge
    assert all(name not in knowledge for name in ("application_submit", "appointment_book", "complaint_create"))
    assert "application_submit" in application and "appointment_book" not in application
    assert "appointment_book" in booking and "application_submit" not in booking
    assert all(name not in conversation for name in ("knowledge_search", "application_submit", "appointment_book"))


def test_transaction_without_channel_is_stopped_before_agent_planning():
    class MustNotRunAgent:
        called = False

        def decide(self, _state, _definitions):
            self.called = True
            raise AssertionError("clarification must stop before domain agent planning")

    agent = MustNotRunAgent()
    workflow = GovServiceAgentWorkflow(
        agent=agent,
        intent_router=route(
            domain="application",
            operation="start",
            item_name="取水许可",
            requested_channel="unspecified",
        ),
    )
    try:
        response = workflow.invoke(
            ChatRequest(query="我想办理取水许可", user_id=1, session_id="routing-clarification")
        )
    finally:
        workflow.close()

    assert agent.called is False
    assert response.route == "clarification"
    assert response.answer == "请选择线上申报或线下预约。"
    assert {action.type for action in response.suggested_actions} == {"select_service_channel"}
    assert {action.payload["channel"] for action in response.suggested_actions} == {"online", "offline"}


def test_explicit_domain_switch_cancels_the_previous_form_state():
    workflow = GovServiceAgentWorkflow(
        intent_router=route(
            domain="appointment",
            operation="book",
            item_name="取水许可",
            requested_channel="offline",
        )
    )
    try:
        routed = workflow.intent_route(
            {
                "normalized_query": "改为线下预约取水许可",
                "workflow_gate": {"mode": "route"},
                "dialog_task": {
                    "task_type": "application_submission",
                    "status": "collecting",
                    "item_code": "ITEM-WATER",
                },
                "events": [],
            }
        )
    finally:
        workflow.close()

    assert routed["route"] == "appointment"
    assert routed["dialog_task"]["status"] == "cancelled"
    assert routed["workflow_gate"]["reason"] == "explicit_domain_switch"


def test_state_machine_rejects_cross_task_and_illegal_terminal_transitions():
    current = {
        "task_type": "application_submission",
        "status": "collecting",
        "item_code": "ITEM-WATER",
    }
    _, wrong_tool = transition_dialog_task(
        current,
        {"task_type": "complaint_create", "status": "collecting"},
        tool_name="complaint_create",
    )
    _, terminal_restart = transition_dialog_task(
        {**current, "status": "completed"},
        {"task_type": "application_submission", "status": "collecting", "item_code": "ITEM-WATER"},
        tool_name="application_submit",
    )

    assert wrong_tool == "active_dialog_task_type_changed"
    assert terminal_restart == "illegal_dialog_transition:completed->collecting"
