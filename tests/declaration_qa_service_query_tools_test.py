from pathlib import Path

from declaration_qa.api.schemas import ChatRequest, HumanConfirmationRequest
from declaration_qa.application.agent import AgentDecision
from declaration_qa.application.nodes.agent_runtime import agent_tool_call_denial_reason, unique_new_tool_calls
from declaration_qa.application.tools import (
    AppointmentBookTool,
    AppointmentCancelTool,
    AppointmentHallQueryTool,
    AppointmentListQueryTool,
    AppointmentSlotQueryTool,
    ConsultationRecordQueryTool,
    DeliveryStatusQueryTool,
    PaymentOrderQueryTool,
    PaymentReceiptQueryTool,
    QueueStatusQueryTool,
    ResultCertificateQueryTool,
    SubmissionReceiptQueryTool,
    TicketDetailQueryTool,
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
        AppointmentHallQueryTool(adapter),
        AppointmentSlotQueryTool(adapter),
        AppointmentBookTool(adapter),
        AppointmentCancelTool(adapter),
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
        {
            "item_code": "11340000002985923L200011900100001",
            "hall_code": "AH-ZW-GCJS",
            "service_date": "2026-07-20",
        },
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
        "appointment_hall_query",
        "appointment_slot_query",
        "appointment_book",
        "appointment_cancel",
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
    slot_schema = definitions["appointment_slot_query"]["parameters"]
    assert "hall_code" in slot_schema["required"]
    assert "appointment_hall_query" in slot_schema["properties"]["hall_code"]["description"]


def test_dependent_tool_schemas_require_current_turn_source_results():
    definitions = {
        item["function"]["name"]: item["function"]
        for item in service_query_registry().definitions()
    }

    dependencies = {
        "appointment_hall_query": ("knowledge_search", "item_code"),
        "appointment_slot_query": ("appointment_hall_query", "hall_code"),
        "appointment_book": ("appointment_slot_query", "slot_id"),
        "appointment_cancel": ("appointment_list_query", "appointment_id"),
        "queue_status_query": ("appointment_list_query", "appointment_id"),
        "payment_order_query": ("ticket_detail_query", "application_ids"),
        "submission_receipt_query": ("ticket_detail_query", "application_ids"),
        "result_certificate_query": ("ticket_detail_query", "application_ids"),
        "delivery_status_query": ("ticket_detail_query", "application_ids"),
    }
    for tool_name, (prerequisite, parameter) in dependencies.items():
        definition = definitions[tool_name]
        parameter_description = definition["parameters"]["properties"][parameter]["description"]
        assert prerequisite in definition["description"]
        assert prerequisite in parameter_description
        assert "历史对话" in parameter_description

    receipt = definitions["payment_receipt_query"]
    assert "ticket_detail_query" in receipt["description"]
    assert "payment_order_query" in receipt["description"]
    assert "targets" in receipt["parameters"]["properties"]


def test_service_query_tools_return_realistic_user_scoped_records():
    registry = service_query_registry()
    user = ToolExecutionContext(user_id=1, region_code="340000")

    appointments = registry.execute("appointment_list_query", {}, user)
    halls = registry.execute(
        "appointment_hall_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT"},
        user,
    )
    slots = registry.execute(
        "appointment_slot_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "hall_code": "AH-ZW-01", "service_date": "2026-07-16"},
        user,
    )
    queue = registry.execute("queue_status_query", {"appointment_id": 4101}, user)
    orders = registry.execute("payment_order_query", {"application_ids": [1001, 2002]}, user)
    payment_receipts = registry.execute(
        "payment_receipt_query",
        {"targets": [{"application_id": 1001, "payment_order_id": 5101}]},
        user,
    )
    submission_receipts = registry.execute("submission_receipt_query", {"application_ids": [2002]}, user)
    certificates = registry.execute("result_certificate_query", {"application_ids": [4004]}, user)
    deliveries = registry.execute("delivery_status_query", {"application_ids": [4004]}, user)
    consultations = registry.execute("consultation_record_query", {}, user)

    assert appointments.data["appointments"][0]["appointment_id"] == 4101
    assert halls.data["halls"] == [
        {
            "hall_code": "AH-ZW-01",
            "hall_name": "安徽省政务服务中心",
            "hall_address": "马鞍山路509号",
        }
    ]
    assert "people_ahead" not in appointments.data["appointments"][0]
    assert slots.data["slots"][0]["remaining_count"] == 6
    assert queue.data["queue_status"]["people_ahead"] == 4
    assert orders.data["payment_orders"][0]["status"] == "paid"
    assert payment_receipts.data["payment_receipts"][0]["download_status"] == "available"
    assert submission_receipts.data["submission_receipts"][0]["receipt_type"] == "submission"
    assert certificates.data["certificates"][0]["certificate_name"] == "档案材料接收确认单"
    assert deliveries.data["deliveries"][0]["status"] == "delivered"
    assert consultations.data["consultations"][0]["status"] == "replied"


def test_application_detail_and_payment_order_tools_only_accept_batch_parameters():
    registry = service_query_registry()
    user = ToolExecutionContext(user_id=1, region_code="340000")
    detail_registry = ToolRegistry()
    detail_registry.register(TicketDetailQueryTool(InMemoryBusinessDataAdapter()))

    old_detail = detail_registry.execute(
        "ticket_detail_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT"},
        user,
    )
    old_orders = registry.execute(
        "payment_order_query",
        {"application_id": 1001},
        user,
    )
    old_receipt = registry.execute("payment_receipt_query", {"application_id": 1001}, user)
    old_submission = registry.execute("submission_receipt_query", {"application_id": 2002}, user)
    old_result = registry.execute("result_certificate_query", {"application_id": 4004}, user)
    old_delivery = registry.execute("delivery_status_query", {"application_id": 4004}, user)

    assert old_detail.status == "invalid_arguments"
    assert old_orders.status == "invalid_arguments"
    assert old_receipt.status == "invalid_arguments"
    assert old_submission.status == "invalid_arguments"
    assert old_result.status == "invalid_arguments"
    assert old_delivery.status == "invalid_arguments"


def test_agent_coalesces_multiple_application_batch_calls():
    calls = [
        tool_call("detail-1", "ticket_detail_query", {"item_codes": ["ITEM-A"]}),
        tool_call("detail-2", "ticket_detail_query", {"item_codes": ["ITEM-B", "ITEM-A"]}),
        tool_call("order-1", "payment_order_query", {"application_ids": [11, 12]}),
        tool_call("order-2", "payment_order_query", {"application_ids": [12, 13]}),
        tool_call("receipt-1", "payment_receipt_query", {"targets": [{"application_id": 11}]}),
        tool_call(
            "receipt-2",
            "payment_receipt_query",
            {"targets": [{"application_id": 12, "payment_order_id": 21}]},
        ),
        tool_call("submission-1", "submission_receipt_query", {"application_ids": [11]}),
        tool_call("submission-2", "submission_receipt_query", {"application_ids": [12]}),
        tool_call("result-1", "result_certificate_query", {"application_ids": [11]}),
        tool_call("result-2", "result_certificate_query", {"application_ids": [13]}),
        tool_call("delivery-1", "delivery_status_query", {"application_ids": [12]}),
        tool_call("delivery-2", "delivery_status_query", {"application_ids": [13]}),
    ]

    merged = unique_new_tool_calls(calls, existing_signatures=set())

    assert [call["name"] for call in merged] == [
        "ticket_detail_query",
        "payment_order_query",
        "payment_receipt_query",
        "submission_receipt_query",
        "result_certificate_query",
        "delivery_status_query",
    ]
    assert merged[0]["arguments"] == {"item_codes": ["ITEM-A", "ITEM-B"], "detail_type": "full"}
    assert merged[1]["arguments"] == {"application_ids": [11, 12, 13]}
    assert merged[2]["arguments"] == {
        "targets": [
            {"application_id": 11, "payment_order_id": None},
            {"application_id": 12, "payment_order_id": 21},
        ]
    }
    assert merged[3]["arguments"] == {"application_ids": [11, 12]}
    assert merged[4]["arguments"] == {"application_ids": [11, 13]}
    assert merged[5]["arguments"] == {"application_ids": [12, 13]}


def test_appointment_book_is_idempotent_and_only_decrements_once():
    registry = service_query_registry()
    user = ToolExecutionContext(user_id=1, region_code="340000", session_id="booking-test")

    before = registry.execute(
        "appointment_slot_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "hall_code": "AH-ZW-01", "service_date": "2026-07-16"},
        user,
    )
    first = registry.execute("appointment_book", {"slot_id": 6103, "confirmed": True}, user)
    replay = registry.execute("appointment_book", {"slot_id": 6103, "confirmed": True}, user)
    after = registry.execute(
        "appointment_slot_query",
        {"item_code": "ITEM-RESIDENCE-PERMIT", "hall_code": "AH-ZW-01", "service_date": "2026-07-16"},
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


def test_appointment_cancel_is_idempotent_and_only_releases_slot_once():
    registry = service_query_registry()
    user = ToolExecutionContext(user_id=1, region_code="340000", session_id="cancel-test")

    before = registry.execute(
        "appointment_slot_query",
        {
            "item_code": "11340000002985923L200011900100001",
            "hall_code": "AH-ZW-GCJS",
            "service_date": "2026-07-20",
        },
        user,
    )
    first = registry.execute(
        "appointment_cancel",
        {"appointment_id": 4301, "confirmed": True},
        user,
    )
    replay = registry.execute(
        "appointment_cancel",
        {"appointment_id": 4301, "confirmed": True},
        user,
    )
    after = registry.execute(
        "appointment_slot_query",
        {
            "item_code": "11340000002985923L200011900100001",
            "hall_code": "AH-ZW-GCJS",
            "service_date": "2026-07-20",
        },
        user,
    )

    before_slot = next(item for item in before.data["slots"] if item["slot_id"] == 6201)
    after_slot = next(item for item in after.data["slots"] if item["slot_id"] == 6201)
    assert first.ok is True
    assert first.data["appointment"]["status"] == "cancelled"
    assert first.data["appointment"]["cancellation_status"] == "cancelled"
    assert replay.data["appointment"]["cancellation_status"] == "already_cancelled"
    assert before_slot["remaining_count"] == 7
    assert after_slot["remaining_count"] == 8


def test_appointment_cancel_rejects_checked_in_or_other_users_records():
    registry = service_query_registry()

    checked_in = registry.execute(
        "appointment_cancel",
        {"appointment_id": 4101, "confirmed": True},
        ToolExecutionContext(user_id=1, region_code="340000"),
    )
    other_user = registry.execute(
        "appointment_cancel",
        {"appointment_id": 4306, "confirmed": True},
        ToolExecutionContext(user_id=1, region_code="340000"),
    )

    assert checked_in.status == "cancellation_not_allowed"
    assert other_user.status == "appointment_not_found"


def test_private_service_query_tools_reject_other_users_records():
    registry = service_query_registry()
    other_user = ToolExecutionContext(user_id=999, region_code="340000")

    assert registry.execute("appointment_list_query", {}, other_user).status == "not_found"
    assert registry.execute("queue_status_query", {"appointment_id": 4101}, other_user).status == "not_found"
    assert registry.execute("payment_order_query", {"application_ids": [1001]}, other_user).status == "not_found"
    assert registry.execute("payment_receipt_query", {"targets": [{"application_id": 1001}]}, other_user).status == "not_found"
    assert registry.execute("submission_receipt_query", {"application_ids": [2002]}, other_user).status == "not_found"
    assert registry.execute("result_certificate_query", {"application_ids": [4004]}, other_user).status == "not_found"
    assert registry.execute("delivery_status_query", {"application_ids": [4004]}, other_user).status == "not_found"
    assert registry.execute("consultation_record_query", {}, other_user).status == "not_found"


def test_service_query_policy_has_no_order_dependent_intercepts():
    assert agent_tool_call_denial_reason(
        "queue_status_query", "我的排队情况", arguments={"appointment_id": 4101}
    ) == ""
    assert agent_tool_call_denial_reason(
        "payment_order_query", "我的缴费订单", arguments={"application_ids": [1001]}
    ) == ""
    assert agent_tool_call_denial_reason(
        "payment_receipt_query",
        "我的电子票据",
        arguments={"targets": [{"application_id": 1001, "payment_order_id": 5101}]},
    ) == ""
    assert agent_tool_call_denial_reason(
        "appointment_book",
        "这个时段还有号吗",
        arguments={"slot_id": 6103, "confirmed": True},
    ) == ""
    assert agent_tool_call_denial_reason(
        "appointment_book",
        "请帮我预约这个时段",
        arguments={"slot_id": 6103},
    ) == ""
    assert agent_tool_call_denial_reason(
        "appointment_book",
        "请帮我预约这个时段",
        arguments={"slot_id": 6103, "confirmed": True},
    ) == ""
    assert agent_tool_call_denial_reason(
        "appointment_cancel",
        "这个预约能取消吗",
        arguments={"appointment_id": 4301, "confirmed": True},
    ) == ""
    assert agent_tool_call_denial_reason(
        "appointment_cancel",
        "请取消这个预约",
        arguments={"appointment_id": 4301},
    ) == ""
    assert agent_tool_call_denial_reason(
        "appointment_cancel",
        "请取消这个预约",
        arguments={"appointment_id": 4301, "confirmed": True},
    ) == ""
    assert agent_tool_call_denial_reason(
        "appointment_book",
        "请帮我预约这个时段",
        arguments={"slot_id": 6102, "confirmed": True},
    ) == ""


def test_agent_can_query_queue_directly_without_appointment_list():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [tool_call("queue", "queue_status_query", {"appointment_id": 4101})],
                )
            return AgentDecision("finish", final_answer="已查询排队状态。[业务1]")

    response = DeclarationQAWorkflow(agent=ScriptedAgent()).invoke(
        ChatRequest(query="查询预约4101的排队状态", user_id=1, region_code="340000")
    )

    assert [result["tool_name"] for result in response.tool_results] == ["queue_status_query"]
    assert response.tool_results[0]["ok"] is True
    assert response.tool_results[0]["status"] == "success"


def test_agent_executes_slot_query_and_booking_without_dependency_wave():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [
                        tool_call(
                            "slots",
                            "appointment_slot_query",
                            {
                                "item_code": "ITEM-RESIDENCE-PERMIT",
                                "hall_code": "AH-ZW-01",
                                "service_date": "2026-07-16",
                            },
                        ),
                        tool_call("book", "appointment_book", {"slot_id": 6103, "confirmed": True}),
                    ],
                )
            return AgentDecision("finish", final_answer="已预约成功。[业务2]")

    workflow = DeclarationQAWorkflow(agent=ScriptedAgent())
    session_id = "human-booking-test"
    events = list(workflow.stream(
        ChatRequest(
            query="请帮我预约7月16日11点的居住证办理时段",
            user_id=1,
            region_code="340000",
            session_id=session_id,
        )
    ))
    confirmation = next(item["data"] for item in events if item["event"] == "human_confirmation_required")

    assert confirmation["actions"][0]["tool_name"] == "appointment_book"
    resumed = list(workflow.resume_stream(HumanConfirmationRequest(
        session_id=session_id,
        interrupt_id=confirmation["interrupt_id"],
        user_id=1,
        approved=True,
    )))
    snapshot = workflow._checkpoint_state(user_id=1, session_id=session_id)

    assert any(item["event"] == "done" for item in resumed)
    assert [result["tool_name"] for result in snapshot["tool_results"]] == [
        "appointment_slot_query",
        "appointment_book",
    ]
    assert all(result["ok"] for result in snapshot["tool_results"])
    assert snapshot["tool_results"][1]["data"]["appointment"]["booking_status"] == "created"
    assert snapshot["answer"] == "已预约成功。[业务2]"


def test_agent_executes_appointment_cancellation_after_human_confirmation():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [
                        tool_call(
                            "cancel",
                            "appointment_cancel",
                            {"appointment_id": 4301, "confirmed": True},
                        )
                    ],
                )
            return AgentDecision("finish", final_answer="已取消预约。[业务1]")

    workflow = DeclarationQAWorkflow(agent=ScriptedAgent())
    session_id = "human-cancel-test"
    events = list(workflow.stream(ChatRequest(
        query="请取消取水许可预约",
        user_id=1,
        region_code="340000",
        session_id=session_id,
    )))
    confirmation = next(item["data"] for item in events if item["event"] == "human_confirmation_required")

    assert confirmation["actions"][0]["arguments"] == {"appointment_id": 4301}
    assert workflow.business_adapter.list_appointments(user_id=1, region_code="340000", status="booked")[0]["appointment_id"] == 4301
    blocked_events = list(workflow.stream(ChatRequest(
        query="先回答另一个问题",
        user_id=1,
        region_code="340000",
        session_id=session_id,
    )))
    blocked_confirmation = next(
        item["data"] for item in blocked_events if item["event"] == "human_confirmation_required"
    )
    assert blocked_confirmation["interrupt_id"] == confirmation["interrupt_id"]
    assert not any(item["event"] == "done" for item in blocked_events)
    resumed = list(workflow.resume_stream(HumanConfirmationRequest(
        session_id=session_id,
        interrupt_id=confirmation["interrupt_id"],
        user_id=1,
        approved=True,
    )))
    snapshot = workflow._checkpoint_state(user_id=1, session_id=session_id)

    assert any(item["event"] == "done" for item in resumed)
    assert snapshot["tool_results"][0]["tool_name"] == "appointment_cancel"
    assert snapshot["tool_results"][0]["ok"] is True
    assert snapshot["tool_results"][0]["data"]["appointment"]["status"] == "cancelled"
    assert snapshot["answer"] == "已取消预约。[业务1]"


def test_agent_does_not_cancel_appointment_when_human_rejects():
    class ScriptedAgent:
        def decide(self, state, _definitions):
            if not state.get("tool_results"):
                return AgentDecision(
                    "tools",
                    [tool_call("cancel", "appointment_cancel", {"appointment_id": 4301})],
                )
            return AgentDecision("finish", final_answer="已保留原预约。")

    workflow = DeclarationQAWorkflow(agent=ScriptedAgent())
    session_id = "human-reject-test"
    events = list(workflow.stream(ChatRequest(
        query="取消取水许可预约",
        user_id=1,
        region_code="340000",
        session_id=session_id,
    )))
    confirmation = next(item["data"] for item in events if item["event"] == "human_confirmation_required")
    list(workflow.resume_stream(HumanConfirmationRequest(
        session_id=session_id,
        interrupt_id=confirmation["interrupt_id"],
        user_id=1,
        approved=False,
    )))
    snapshot = workflow._checkpoint_state(user_id=1, session_id=session_id)

    assert snapshot["tool_results"][0]["status"] == "human_rejected"
    booked_ids = {
        item["appointment_id"]
        for item in workflow.business_adapter.list_appointments(user_id=1, region_code="340000", status="booked")
    }
    assert 4301 in booked_ids


def test_agent_executes_business_calls_without_dependency_waves():
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
                            {"item_codes": ["ITEM-HEALTH-CERTIFICATE"]},
                        ),
                        tool_call("orders", "payment_order_query", {"application_ids": [1001]}),
                        tool_call(
                            "receipt",
                            "payment_receipt_query",
                            {"targets": [{"application_id": 1001, "payment_order_id": 5101}]},
                        ),
                    ],
                )
            return AgentDecision("finish", final_answer="已查询到缴费订单和电子票据。[业务3][业务4]")

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
    assert response.answer == "已查询到缴费订单和电子票据。[业务3][业务4]"
