from __future__ import annotations

import pytest

from declaration_qa.application.evidence_registry import build_evidence_registry, extract_business_key
from declaration_qa.memory.evidence import project_historical_messages
from declaration_qa.rag.prompting import format_source_payload, format_tool_result_payload


def knowledge_source(text: str) -> dict[str, object]:
    return {
        "chunk_id": "chunk-law-8",
        "doc_id": "doc-adoption-rule",
        "title": "外国人在中华人民共和国收养子女登记办法",
        "evidence_text": text,
        "score": 0.9,
    }


def test_evidence_payloads_publish_stable_inline_identifiers():
    source_payload = format_source_payload([knowledge_source("原文内容")])
    tool_payload = format_tool_result_payload(
        [
            {"tool_name": "knowledge_search", "ok": True, "data": {}},
            {
                "tool_name": "ticket_list_query",
                "call_id": "call-ticket-list",
                "ok": True,
                "status": "success",
                "data": {"items": [{"item_code": "ITEM-001"}]},
            },
        ]
    )

    assert "[证据1]" in source_payload
    assert "块标识：chunk-law-8" in source_payload
    assert '"evidence_id": "业务1"' in tool_payload
    assert '"call_id": "call-ticket-list"' in tool_payload
    assert '"business_key": {"item_code": ["ITEM-001"]}' in tool_payload


def test_evidence_registry_maps_display_ids_to_underlying_ids():
    registry = build_evidence_registry(
        [knowledge_source("原文内容")],
        [
            {
                "tool_name": "ticket_detail_query",
                "call_id": "call-ticket-detail",
                "ok": True,
                "data": {
                    "applications": [
                        {"application_id": 15},
                        {"application_id": 14},
                        {"application_id": 15},
                    ]
                },
            }
        ],
    )

    assert registry["knowledge"] == [{"evidence_id": "证据1", "chunk_id": "chunk-law-8"}]
    assert registry["business"] == [
        {
            "evidence_id": "业务1",
            "call_id": "call-ticket-detail",
            "tool_name": "ticket_detail_query",
            "business_key": {"application_id": [15, 14]},
        }
    ]


def test_failed_business_result_does_not_consume_a_business_number():
    tool_results = [
        {
            "tool_name": "queue_status_query",
            "call_id": "call-failed",
            "ok": False,
            "status": "not_found",
            "data": {"appointment_id": 1},
        },
        {
            "tool_name": "queue_status_query",
            "call_id": "call-success",
            "ok": True,
            "status": "success",
            "data": {"appointment_id": 1, "queue_status": {"waiting_count": 4}},
        },
    ]

    payload = format_tool_result_payload(tool_results)
    assert '"evidence_id": "业务1"' in payload
    assert '"call_id": "call-success"' in payload
    assert "call-failed" not in payload


@pytest.mark.parametrize(
    ("tool_name", "field", "value"),
    [
        ("ticket_list_query", "item_code", "ITEM-001"),
        ("ticket_detail_query", "application_id", 15),
        ("appointment_list_query", "appointment_id", 4101),
        ("appointment_slot_query", "slot_id", 6101),
        ("appointment_book", "appointment_id", 4102),
        ("queue_status_query", "appointment_id", 4101),
        ("payment_order_query", "payment_order_id", 5101),
        ("payment_receipt_query", "payment_receipt_id", 5201),
        ("submission_receipt_query", "submission_receipt_id", 5301),
        ("result_certificate_query", "result_certificate_id", 5401),
        ("delivery_status_query", "delivery_id", 5501),
        ("consultation_record_query", "consultation_id", 5601),
    ],
)
def test_each_business_tool_uses_its_declared_primary_key(tool_name: str, field: str, value: object):
    assert extract_business_key(tool_name, {"records": [{field: value}]}) == {field: [value]}


def test_historical_assistant_text_is_marked_untrusted_and_tool_protocol_is_removed():
    messages = [
        {"id": "user-1", "role": "user", "content": "融合协议是什么"},
        {"id": "assistant-1", "role": "assistant", "content": "这是法定文书。"},
        {
            "id": "assistant-2",
            "role": "assistant",
            "content": "准备查询",
            "tool_calls": [{"id": "call-1", "name": "knowledge_search", "args": {}}],
        },
        {
            "id": "tool-1",
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "knowledge_search",
            "content": {
                "tool_name": "knowledge_search",
                "call_id": "call-1",
                "ok": True,
                "data": {
                    "sources": [
                        {
                            "chunk_id": "chunk-1",
                            "evidence_text": "登记机关办理收养登记。",
                        }
                    ]
                },
            },
        },
    ]

    isolated = project_historical_messages(messages)

    assert isolated[0]["content"] == "融合协议是什么"
    assert "这是法定文书" in isolated[1]["content"]
    assert "仅用于理解指代，不得作为事实证据" in isolated[1]["content"]
    assert len(isolated) == 2
    assert all(item.get("role") != "tool" for item in isolated)
    assert all(not item.get("tool_calls") for item in isolated)


def test_historical_knowledge_and_business_tools_are_removed_for_new_turn():
    messages = []
    for turn in range(1, 5):
        knowledge_call_id = f"knowledge-call-{turn}"
        business_call_id = f"business-call-{turn}"
        messages.extend(
            [
                {"id": f"user-{turn}", "role": "user", "content": f"第{turn}轮"},
                {
                    "id": f"assistant-{turn}",
                    "role": "assistant",
                    "content": "查询",
                    "tool_calls": [
                        {"id": knowledge_call_id, "name": "knowledge_search", "args": {}},
                        {"id": business_call_id, "name": "queue_status_query", "args": {}},
                    ],
                },
                {
                    "id": f"tool-{turn}",
                    "role": "tool",
                    "tool_call_id": knowledge_call_id,
                    "name": "knowledge_search",
                    "content": {
                        "tool_name": "knowledge_search",
                        "call_id": knowledge_call_id,
                        "ok": True,
                        "data": {
                            "sources": [
                                {
                                    "chunk_id": f"chunk-{turn}",
                                    "evidence_text": f"第{turn}轮证据原文",
                                }
                            ]
                        },
                    },
                },
                {
                    "id": f"business-tool-{turn}",
                    "role": "tool",
                    "tool_call_id": business_call_id,
                    "name": "queue_status_query",
                    "content": {
                        "tool_name": "queue_status_query",
                        "call_id": business_call_id,
                        "ok": True,
                        "data": {"appointment_id": turn, "waiting_count": turn},
                    },
                },
                {
                    "id": f"answer-{turn}",
                    "role": "assistant",
                    "content": f"第{turn}轮业务结论。[证据1][业务1]",
                },
            ]
        )

    projected = project_historical_messages(messages, keep_recent_user_turns=3)
    serialized = str(projected)

    for turn in range(1, 5):
        assert f"knowledge-call-{turn}" not in serialized
        assert f"business-call-{turn}" not in serialized
        assert f"第{turn}轮证据原文" not in serialized
    assert "第1轮业务结论" not in serialized
    for turn in range(2, 5):
        assert f"第{turn}轮业务结论" in serialized


def test_knowledge_evidence_is_removed_even_before_rolling_summary():
    messages = [
        {"id": "user-1", "role": "user", "content": "第一轮问题"},
        {
            "id": "assistant-tool-1",
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "knowledge-call-1", "name": "knowledge_search", "args": {}}
            ],
        },
        {
            "id": "knowledge-tool-1",
            "role": "tool",
            "tool_call_id": "knowledge-call-1",
            "name": "knowledge_search",
            "content": {
                "tool_name": "knowledge_search",
                "call_id": "knowledge-call-1",
                "ok": True,
                "data": {
                    "sources": [
                        {"chunk_id": "chunk-1", "evidence_text": "第一轮证据原文"}
                    ]
                },
            },
        },
        {"id": "answer-1", "role": "assistant", "content": "第一轮回答"},
        {"id": "user-2", "role": "user", "content": "第二轮问题"},
    ]

    projected = project_historical_messages(
        messages,
        keep_recent_user_turns=0,
        memory_context={},
    )

    serialized = str(projected)
    assert "knowledge-call-1" not in serialized
    assert "chunk-1" not in serialized
    assert "第一轮证据原文" not in serialized
