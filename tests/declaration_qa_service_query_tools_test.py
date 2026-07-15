from pathlib import Path

from declaration_qa.api.schemas import ChatRequest
from declaration_qa.application.agent import AgentDecision
from declaration_qa.application.nodes.agent_runtime import agent_tool_call_denial_reason
from declaration_qa.application.tools import (
    AppointmentBookTool,
    AppointmentListQueryTool,
    AppointmentSlotQueryTool,
    ConsultationRecordQueryTool,
    DeliveryStatusQueryTool,
    PaymentOrderQueryTool,
    PaymentReceiptQueryTool,
    QueueStatusQueryTool,
    ResultCertificateQueryTool,
    SubmissionReceiptQueryTool,
    ToolExecutionContext,
    ToolRegistry,
)
from declaration_qa.application.workflow import DeclarationQAWorkflow
from declaration_qa.core.business import InMemoryBusinessDataAdapter


def service_query_registry(adapter=None):
    adapter = adapter or InMemoryBusinessDataAdapter()
    registry = ToolRegistry()
    for tool in [
        AppointmentListQueryTool(adapter),
        AppointmentSlotQueryTool(adapter),
        AppointmentBookTool(adapter),
        QueueStatusQueryTool(adapter),
        PaymentOrderQueryTool(adapter),
        PaymentReceiptQueryTool(adapter),
        SubmissionReceiptQueryTool(adapter),
        ResultCertificateQueryTool(adapter),
        DeliveryStatusQueryTool(adapter),
        ConsultationRecordQueryTool(adapter),
    ]:
        registry.register(tool)
    return registry


def tool_call(call_id, name, arguments=None):
    return {"id": call_id, "name": name, "arguments": arguments or {}}


def test_combined_postgresql_schema_contains_service_query_tables():
    schema = Path("sql/postgresql/declaration_qa_postgresql_all.sql").read_text(encoding="utf-8")

    for table in [
        "business_appointment_slots",
        "business_appointments",
        "business_payment_orders",
        "business_payment_receipts",
        "business_submission_receipts",
        "business_result_certificates",
        "business_deliveries",
        "business_consultations",
    ]:
        assert f"CREATE TABLE IF NOT EXISTS declaration_qa.{table}" in schema


def test_submitted_document_mock_data_keeps_official_item_mappings():
    migration = Path("sql/postgresql/011_submitted_document_mock_appointments.sql").read_text(encoding="utf-8")
    source_dir = Path("materials/curated/declaration_qa/2026-06-15-ahzwfw-real-40-topics")
    mappings = {
        "item_28.md": ("item_28_official_ahzwfw", "11340000002985923L200011900100001", "取水许可"),
        "item_30.md": ("item_30_official_ahzwfw", "11340000002985990J200016407900001", "草原防火期内在草原上进行爆破、勘察和施工等活动审批"),
        "high_risk_sports_event_permit.md": (
            "high_risk_sports_event_permit_official_ahzwfw",
            "11340000002985974W234013300500001",
            "举办高危险性体育赛事活动许可",
        ),
        "foreign_hmt_overseas_chinese_marriage_registration.md": (
            "foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw",
            "11340000002986424J200071100100002",
            "涉外、涉港澳台、涉华侨结婚登记",
        ),
        "fire_accident_identification_review.md": (
            "fire_accident_identification_review_official_ahzwfw",
            "11340000MB1C86468R234070901300001",
            "火灾事故认定复核",
        ),
    }

    for filename, (manifest_id, item_code, item_name) in mappings.items():
        document = (source_dir / filename).read_text(encoding="utf-8")
        assert manifest_id in migration
        assert item_code in document and item_code in migration
        assert item_name in document and item_name in migration


def test_document_derived_mock_slots_and_appointments_are_queryable():
    registry = service_query_registry()
    user = ToolExecutionContext(user_id=1, region_code="340000")

    slots = registry.execute(
        "appointment_slot_query",
        {"item_code": "11340000002985923L200011900100001", "service_date": "2026-07-20"},
        user,
    )
    appointments = registry.execute("appointment_list_query", {"status": "all"}, user)

    assert [slot["remaining_count"] for slot in slots.data["slots"]] == [7, 0, 10]
    assert all(slot["source_manifest_id"] == "item_28_official_ahzwfw" for slot in slots.data["slots"])
    assert any(
        item["appointment_id"] == 4301 and item["hall_code"] == "AH-ZW-GCJS"
        for item in appointments.data["appointments"]
    )
    assert all(item["appointment_id"] != 4306 for item in appointments.data["appointments"])


def test_service_query_tool_schemas_never_accept_user_identity():
    definitions = {
        item["function"]["name"]: item["function"]
        for item in service_query_registry().definitions()
    }

    assert set(definitions) == {
        "appointment_list_query",
        "appointment_slot_query",
        "appointment_book",
        "queue_status_query",
        "payment_order_query",
        "payment_receipt_query",
        "submission_receipt_query",
        "result_certificate_query",
        "delivery_status_query",
        "consultation_record_query",
    }
    for definition in definitions.values():
        properties = definition["parameters"].get("properties", {})
        assert "user_id" not in properties
        assert "region_code" not in properties


def test_service_query_tools_return_realistic_user_scoped_records():
    registry = service_query_registry()
    user = ToolExecutionContext(user_id=1, region_code="340000")

    appointments = registry.execute("appointment_list_query", {}, user)
    slots = registry.execute(
        "appointment_slot_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "service_date": "2026-07-16"},
        user,
    )
    queue = registry.execute("queue_status_query", {"appointment_id": 4101}, user)
    orders = registry.execute("payment_order_query", {"application_id": 1001}, user)
    payment_receipts = registry.execute(
        "payment_receipt_query",
        {"application_id": 1001, "payment_order_id": 5101},
        user,
    )
    submission_receipts = registry.execute("submission_receipt_query", {"application_id": 2002}, user)
    certificates = registry.execute("result_certificate_query", {"application_id": 4004}, user)
    deliveries = registry.execute("delivery_status_query", {"application_id": 4004}, user)
    consultations = registry.execute("consultation_record_query", {}, user)

    assert appointments.data["appointments"][0]["appointment_id"] == 4101
    assert "people_ahead" not in appointments.data["appointments"][0]
    assert slots.data["slots"][0]["remaining_count"] == 6
    assert queue.data["queue_status"]["people_ahead"] == 4
    assert orders.data["payment_orders"][0]["status"] == "paid"
    assert payment_receipts.data["payment_receipts"][0]["download_status"] == "available"
    assert submission_receipts.data["submission_receipts"][0]["receipt_type"] == "submission"
    assert certificates.data["certificates"][0]["certificate_name"] == "档案材料接收确认单"
    assert deliveries.data["deliveries"][0]["status"] == "delivered"
    assert consultations.data["consultations"][0]["status"] == "replied"


def test_appointment_book_is_idempotent_and_only_decrements_once():
    registry = service_query_registry()
    user = ToolExecutionContext(user_id=1, region_code="340000", session_id="booking-test")

    before = registry.execute(
        "appointment_slot_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "service_date": "2026-07-16"},
        user,
    )
    first = registry.execute("appointment_book", {"slot_id": 6103, "confirmed": True}, user)
    replay = registry.execute("appointment_book", {"slot_id": 6103, "confirmed": True}, user)
    after = registry.execute(
        "appointment_slot_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "service_date": "2026-07-16"},
        user,
    )

    before_slot = next(item for item in before.data["slots"] if item["slot_id"] == 6103)
    after_slot = next(item for item in after.data["slots"] if item["slot_id"] == 6103)
    assert first.ok is True
    assert first.data["appointment"]["booking_status"] == "created"
    assert replay.data["appointment"]["booking_status"] == "idempotent_replay"
    assert replay.data["appointment"]["appointment_id"] == first.data["appointment"]["appointment_id"]
    assert before_slot["remaining_count"] == 10
    assert after_slot["remaining_count"] == 9


def test_appointment_book_rejects_full_or_other_region_slots():
    registry = service_query_registry()

    full = registry.execute(
        "appointment_book",
        {"slot_id": 6102, "confirmed": True},
        ToolExecutionContext(user_id=1, region_code="340000", session_id="full-slot"),
    )
    other_region = registry.execute(
        "appointment_book",
        {"slot_id": 6103, "confirmed": True},
        ToolExecutionContext(user_id=1, region_code="341000", session_id="other-region"),
    )

    assert full.status == "slot_unavailable"
    assert other_region.status == "slot_not_found"


def test_private_service_query_tools_reject_other_users_records():
    registry = service_query_registry()
    other_user = ToolExecutionContext(user_id=999, region_code="340000")

    assert registry.execute("appointment_list_query", {}, other_user).status == "not_found"
    assert registry.execute("queue_status_query", {"appointment_id": 4101}, other_user).status == "not_found"
    assert registry.execute("payment_order_query", {"application_id": 1001}, other_user).status == "not_found"
    assert registry.execute("payment_receipt_query", {"application_id": 1001}, other_user).status == "not_found"
    assert registry.execute("submission_receipt_query", {"application_id": 2002}, other_user).status == "not_found"
    assert registry.execute("result_certificate_query", {"application_id": 4004}, other_user).status == "not_found"
    assert registry.execute("delivery_status_query", {"application_id": 4004}, other_user).status == "not_found"
    assert registry.execute("consultation_record_query", {}, other_user).status == "not_found"


def test_service_query_policy_requires_observed_owned_identifiers():
    appointment_results = [
        {
            "tool_name": "appointment_list_query",
            "ok": True,
            "data": {"appointments": [{"appointment_id": 4101}]},
        }
    ]
    application_results = [
        {
            "tool_name": "ticket_detail_query",
            "ok": True,
            "data": {"applications": [{"application_id": 1001}]},
        }
    ]
    payment_results = [
        *application_results,
        {
            "tool_name": "payment_order_query",
            "ok": True,
            "data": {
                "application_id": 1001,
                "payment_orders": [{"application_id": 1001, "payment_order_id": 5101}],
            },
        },
    ]
    slot_results = [
        {
            "tool_name": "appointment_slot_query",
            "ok": True,
            "data": {
                "slots": [
                    {"slot_id": 6102, "status": "full", "remaining_count": 0},
                    {"slot_id": 6103, "status": "available", "remaining_count": 10},
                ]
            },
        }
    ]

    assert "requires a successful" in agent_tool_call_denial_reason(
        "queue_status_query", "我的排队情况", arguments={"appointment_id": 4101}, prior_results=[]
    )
    assert agent_tool_call_denial_reason(
        "queue_status_query", "我的排队情况", arguments={"appointment_id": 4101}, prior_results=appointment_results
    ) == ""
    assert "was not returned" in agent_tool_call_denial_reason(
        "queue_status_query", "我的排队情况", arguments={"appointment_id": 4201}, prior_results=appointment_results
    )
    assert "requires a successful" in agent_tool_call_denial_reason(
        "payment_order_query", "我的缴费订单", arguments={"application_id": 1001}, prior_results=[]
    )
    assert agent_tool_call_denial_reason(
        "payment_order_query", "我的缴费订单", arguments={"application_id": 1001}, prior_results=application_results
    ) == ""
    assert agent_tool_call_denial_reason(
        "payment_receipt_query",
        "我的电子票据",
        arguments={"application_id": 1001, "payment_order_id": 5101},
        prior_results=payment_results,
    ) == ""
    assert "was not returned" in agent_tool_call_denial_reason(
        "payment_receipt_query",
        "我的电子票据",
        arguments={"application_id": 1001, "payment_order_id": 9999},
        prior_results=payment_results,
    )
    assert "explicit user confirmation" in agent_tool_call_denial_reason(
        "appointment_book",
        "这个时段还有号吗",
        arguments={"slot_id": 6103, "confirmed": True},
        prior_results=slot_results,
    )
    assert "confirmed=true" in agent_tool_call_denial_reason(
        "appointment_book",
        "请帮我预约这个时段",
        arguments={"slot_id": 6103},
        prior_results=slot_results,
    )
    assert agent_tool_call_denial_reason(
        "appointment_book",
        "请帮我预约这个时段",
        arguments={"slot_id": 6103, "confirmed": True},
        prior_results=slot_results,
    ) == ""
    assert "was not returned as available" in agent_tool_call_denial_reason(
        "appointment_book",
        "请帮我预约这个时段",
        arguments={"slot_id": 6102, "confirmed": True},
        prior_results=slot_results,
    )


def test_agent_queries_slot_before_booking_in_same_decision():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [
                        tool_call(
                            "slots",
                            "appointment_slot_query",
                            {"item_code": "ITEM-RESIDENCE-PERMIT", "service_date": "2026-07-16"},
                        ),
                        tool_call("book", "appointment_book", {"slot_id": 6103, "confirmed": True}),
                    ],
                )
            return AgentDecision("finish", final_answer="已预约成功。")

    response = DeclarationQAWorkflow(agent=ScriptedAgent()).invoke(
        ChatRequest(query="请帮我预约7月16日11点的居住证办理时段", user_id=1, region_code="340000")
    )

    assert [result["tool_name"] for result in response.tool_results] == [
        "appointment_slot_query",
        "appointment_book",
    ]
    assert all(result["ok"] for result in response.tool_results)
    assert response.tool_results[1]["data"]["appointment"]["booking_status"] == "created"
    assert response.answer == "已预约成功。"


def test_agent_executes_multi_level_business_dependencies_in_order():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [
                        tool_call("tickets", "ticket_list_query"),
                        tool_call(
                            "detail",
                            "ticket_detail_query",
                            {"item_code": "ITEM-HEALTH-CERTIFICATE", "application_id": 1001},
                        ),
                        tool_call("orders", "payment_order_query", {"application_id": 1001}),
                        tool_call(
                            "receipt",
                            "payment_receipt_query",
                            {"application_id": 1001, "payment_order_id": 5101},
                        ),
                    ],
                )
            return AgentDecision("finish", final_answer="已查询到缴费订单和电子票据。")

    response = DeclarationQAWorkflow(agent=ScriptedAgent()).invoke(
        ChatRequest(query="查询我的健康证缴费订单和电子票据", user_id=1, region_code="340000")
    )

    assert [result["tool_name"] for result in response.tool_results] == [
        "ticket_list_query",
        "ticket_detail_query",
        "payment_order_query",
        "payment_receipt_query",
    ]
    assert all(result["ok"] for result in response.tool_results)
    assert response.answer == "已查询到缴费订单和电子票据。"
