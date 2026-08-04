from gov_service_agent.api.schemas import ChatRequest
from gov_service_agent.application.service_channel import (
    OFFLINE_CHANNEL,
    ONLINE_CHANNEL,
    detect_service_channel,
    filter_tool_definitions_for_channel,
    requests_channel_switch,
    requires_service_channel_clarification,
)
from gov_service_agent.application.workflow import GovServiceAgentWorkflow


def test_service_channel_detection_distinguishes_online_offline_and_ambiguous_requests():
    assert detect_service_channel("我要线上办理取水许可") == ONLINE_CHANNEL
    assert detect_service_channel("我要去窗口办理取水许可") == OFFLINE_CHANNEL
    assert detect_service_channel("我想预约取水许可") == OFFLINE_CHANNEL
    assert detect_service_channel("我想办理取水许可") == ""
    assert requires_service_channel_clarification("我想办理取水许可", "") is True
    assert requires_service_channel_clarification("取水许可需要什么材料", "") is False


def test_channel_filters_hide_the_incompatible_write_chain_from_the_model():
    definitions = [
        {"type": "function", "function": {"name": "knowledge_search"}},
        {"type": "function", "function": {"name": "application_submit"}},
        {"type": "function", "function": {"name": "appointment_hall_query"}},
        {"type": "function", "function": {"name": "appointment_slot_query"}},
        {"type": "function", "function": {"name": "appointment_book"}},
    ]

    online = filter_tool_definitions_for_channel(definitions, ONLINE_CHANNEL)
    offline = filter_tool_definitions_for_channel(definitions, OFFLINE_CHANNEL)
    online_names = {item["function"]["name"] for item in online}
    offline_names = {item["function"]["name"] for item in offline}

    assert "application_submit" in online_names
    assert not {"appointment_hall_query", "appointment_slot_query", "appointment_book"} & online_names
    assert "application_submit" not in offline_names
    assert {"appointment_hall_query", "appointment_slot_query", "appointment_book"} <= offline_names


def test_ambiguous_transaction_finishes_with_channel_question_without_calling_agent():
    class MustNotRunAgent:
        called = False

        def decide(self, _state, _definitions):
            self.called = True
            raise AssertionError("ambiguous channel must be handled before model planning")

    agent = MustNotRunAgent()
    workflow = GovServiceAgentWorkflow(agent=agent)
    try:
        response = workflow.invoke(
            ChatRequest(
                query="我想办理取水许可",
                user_id=1,
                session_id="service-channel-clarification",
            )
        )
    finally:
        workflow.close()

    assert agent.called is False
    assert "请选择办理方式" in response.answer
    assert {action.type for action in response.suggested_actions} == {"select_service_channel"}
    assert {action.payload["channel"] for action in response.suggested_actions} == {"online", "offline"}
    assert response.human_confirmation == {}
    assert response.dialog_task == {}


def test_explicit_offline_switch_cancels_inflight_online_slot_collection():
    assert requests_channel_switch("改为线下办理", OFFLINE_CHANNEL) is True
    workflow = GovServiceAgentWorkflow()
    try:
        prepared = workflow.agent_prepare(
            {
                "query": "改为线下办理",
                "normalized_query": "改为线下办理",
                "user_id": 1,
                "session_id": "service-channel-switch",
                "region_code": "340000",
                "dialog_task": {
                    "task_type": "application_submission",
                    "status": "collecting",
                    "item_code": "ITEM-WATER",
                    "values": {"applicant_name": "测试申请人"},
                },
                "agent_messages": [],
                "events": [],
            }
        )
    finally:
        workflow.close()

    assert prepared["service_channel"] == OFFLINE_CHANNEL
    assert prepared["dialog_task"] == {}
