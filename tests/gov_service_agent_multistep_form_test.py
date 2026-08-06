from gov_service_agent.application.tools.registry import ToolExecutionContext
from gov_service_agent.application.tools.submissions import ApplicationSubmitTool
from gov_service_agent.core.business import InMemoryBusinessDataAdapter


def tool_context(task=None, call_id="step-test"):
    return ToolExecutionContext(
        user_id=1,
        region_code="340000",
        session_id="multistep-form-test",
        metadata={
            "state": {"dialog_task": task or {}},
            "tool_call_id": call_id,
        },
    )


def test_each_application_schema_exposes_ordered_steps_with_descriptions():
    adapter = InMemoryBusinessDataAdapter()

    residence = adapter.get_application_form_schema(item_code="ITEM-RESIDENCE-PERMIT")
    health = adapter.get_application_form_schema(item_code="ITEM-HEALTH-CERTIFICATE")
    water = adapter.get_application_form_schema(item_code="11340000002985923L200011900100001")

    assert [step["id"] for step in residence["steps"]] == ["applicant_identity", "residence_details"]
    assert len(health["steps"]) == 3
    assert len(water["steps"]) == 3
    for schema in (residence, health, water):
        assert all(step["title"] and step["description"] and step["fields"] for step in schema["steps"])


def test_application_submission_advances_one_step_and_rejects_out_of_order_step():
    adapter = InMemoryBusinessDataAdapter()
    tool = ApplicationSubmitTool(adapter)
    first = tool.confirmation_preview(
        {
            "item_code": "ITEM-RESIDENCE-PERMIT",
            "values": {"姓名": "张三", "手机号": "13800138000"},
        },
        tool_context(),
    )

    assert first["ready"] is False
    assert first["dialog_task"]["current_step_id"] == "applicant_identity"

    stale = tool.confirmation_preview(
        {"step_id": "residence_details", "values": {"居住地址": "合肥市包河区示例路1号"}},
        tool_context(first["dialog_task"], "stale-step"),
    )
    assert stale["error"] == "stale_or_out_of_order_step"

    second = tool.confirmation_preview(
        {
            "step_id": "applicant_identity",
            "values": {"身份证号": "340101199001011234"},
        },
        tool_context(first["dialog_task"], "identity-step"),
    )
    assert second["ready"] is False
    assert second["dialog_task"]["current_step_id"] == "residence_details"
    assert second["dialog_task"]["completed_step_ids"] == ["applicant_identity"]


def test_final_step_is_the_only_step_that_can_be_ready_for_confirmation():
    adapter = InMemoryBusinessDataAdapter()
    tool = ApplicationSubmitTool(adapter)
    first = tool.confirmation_preview(
        {
            "item_code": "ITEM-RESIDENCE-PERMIT",
            "values": {
                "姓名": "张三",
                "手机号": "13800138000",
                "身份证号": "340101199001011234",
            },
        },
        tool_context(),
    )
    assert first["ready"] is False

    second = tool.confirmation_preview(
        {"step_id": "applicant_identity", "values": {}},
        tool_context(first["dialog_task"], "identity-step"),
    )
    final = tool.confirmation_preview(
        {
            "step_id": "residence_details",
            "values": {"居住地址": "合肥市包河区示例路1号", "证明编号": "RES-2026-001"},
        },
        tool_context(second["dialog_task"], "residence-step"),
    )
    assert final["ready"] is True
    assert final["dialog_task"]["status"] == "ready"
    assert final["dialog_task"]["completed_step_ids"] == ["applicant_identity", "residence_details"]
