from __future__ import annotations

import pytest

from declaration_qa.application.evidence_grounding import (
    isolate_untrusted_assistant_history,
    validate_grounded_answer,
)
from declaration_qa.application.evidence_registry import build_evidence_registry, extract_business_key
from declaration_qa.memory.evidence import project_historical_messages, recent_tool_context
from declaration_qa.rag.prompting import format_source_payload, format_tool_result_payload


def knowledge_source(text: str) -> dict[str, object]:
    return {
        "chunk_id": "chunk-law-8",
        "doc_id": "doc-adoption-rule",
        "title": "外国人在中华人民共和国收养子女登记办法",
        "evidence_text": text,
        "score": 0.9,
    }


def test_legal_reference_can_use_a_cited_summary_without_an_exact_quote():
    state = {
        "query": "登记办法第八条是怎么规定的",
        "retrieved_sources": [
            knowledge_source("第八条 外国收养人来华办理收养登记，应当亲自来华办理。")
        ],
        "tool_results": [{"tool_name": "knowledge_search", "ok": True, "data": {}}],
    }

    decision = validate_grounded_answer(
        state,
        "《外国人在中华人民共和国收养子女登记办法》第八条规定了外国收养人来华办理登记的要求。[证据1]",
    )

    assert decision["ok"] is True


def test_external_fact_without_an_inline_evidence_marker_is_not_sentence_checked():
    state = {
        "query": "健康证需要哪些材料",
        "retrieved_sources": [knowledge_source("申请健康证需要身份证明和体检材料。")],
        "tool_results": [{"tool_name": "knowledge_search", "ok": True, "data": {}}],
    }

    decision = validate_grounded_answer(state, "申请健康证需要身份证明和体检材料。")

    assert decision["ok"] is True
    assert decision["reason"] == ""


def test_uncited_intro_and_advice_do_not_block_valid_cited_content():
    state = {
        "query": "健康证需要哪些材料",
        "retrieved_sources": [knowledge_source("申请健康证需要身份证明和体检材料。")],
        "tool_results": [{"tool_name": "knowledge_search", "ok": True, "data": {}}],
    }

    decision = validate_grounded_answer(
        state,
        "以下是申请材料：\n- 身份证明和体检材料。[证据1]\n如需帮助，可以继续告诉我。",
    )

    assert decision["ok"] is True
    assert "uncited_fact_units" not in decision["checks"]


def test_uncited_business_summary_and_advice_do_not_block_valid_cited_items():
    state = {
        "query": "我有哪些事项",
        "retrieved_sources": [],
        "tool_results": [
            {
                "tool_name": "ticket_list_query",
                "call_id": "ticket-list-1",
                "ok": True,
                "data": {
                    "items": [
                        {"item_code": "ITEM-WATER", "item_name": "取水许可"}
                    ]
                },
            }
        ],
    }

    decision = validate_grounded_answer(
        state,
        "您当前共办理过1项政务事项：\n- 取水许可。[业务1]\n如需查看进度，请告诉我事项名称。",
    )

    assert decision["ok"] is True


def test_external_question_without_trusted_evidence_is_rejected():
    decision = validate_grounded_answer(
        {"query": "融合协议是不是法定材料", "retrieved_sources": [], "tool_results": []},
        "融合协议是法定材料。",
    )

    assert decision["ok"] is False
    assert decision["reason"] == "no_trusted_evidence"


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
    decision = validate_grounded_answer(
        {"query": "我的排队情况", "retrieved_sources": [], "tool_results": tool_results},
        "当前前方等待人数为4人。[业务1]",
    )

    assert '"evidence_id": "业务1"' in payload
    assert '"call_id": "call-success"' in payload
    assert "call-failed" not in payload
    assert decision["ok"] is True
    assert decision["evidence_registry"]["business"][0]["call_id"] == "call-success"


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


def test_historical_assistant_text_is_marked_untrusted_and_tool_protocol_is_preserved():
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

    isolated = isolate_untrusted_assistant_history(messages)

    assert isolated[0]["content"] == "融合协议是什么"
    assert "这是法定文书" in isolated[1]["content"]
    assert "仅用于理解指代，不得作为事实证据" in isolated[1]["content"]
    assert isolated[2]["content"] == ""
    assert isolated[2]["tool_calls"][0]["id"] == "call-1"
    assert isolated[3]["content"]["data"]["sources"][0]["evidence_text"] == "登记机关办理收养登记。"


def test_knowledge_evidence_survives_three_turns_while_business_results_do_not():
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

    assert "knowledge-call-1" in serialized
    assert "chunk-1" in serialized
    assert "第1轮证据原文" in serialized
    assert "business-call-1" not in serialized
    assert "第1轮业务结论" not in serialized
    for turn in range(2, 5):
        assert f"knowledge-call-{turn}" in serialized
        assert f"business-call-{turn}" in serialized
        assert f"第{turn}轮证据原文" in serialized
        assert f"第{turn}轮业务结论" in serialized


def test_knowledge_evidence_is_removed_after_rolling_summary_covers_it():
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
        memory_context={
            "running_summary": {
                "summarized_message_ids": {
                    "user-1",
                    "assistant-tool-1",
                    "knowledge-tool-1",
                    "answer-1",
                },
                "last_summarized_message_id": "answer-1",
            }
        },
    )

    serialized = str(projected)
    assert "knowledge-call-1" not in serialized
    assert "chunk-1" not in serialized
    assert "第一轮证据原文" not in serialized


def test_recent_tool_context_uses_full_text_and_never_uses_reference_only_records():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "knowledge-full",
            "name": "knowledge_search",
            "content": {
                "tool_name": "knowledge_search",
                "call_id": "knowledge-full",
                "ok": True,
                "data": {
                    "sources": [
                        {
                            "chunk_id": "chunk-full",
                            "title": "完整证据",
                            "evidence_text": "应当交验身份证明原件。",
                        }
                    ]
                },
            },
        },
        {
            "role": "tool",
            "tool_call_id": "knowledge-reference",
            "name": "knowledge_search",
            "content": {
                "tool_name": "knowledge_search",
                "call_id": "knowledge-reference",
                "ok": True,
                "memory_trust": "reference_only",
                "data": {"source_refs": [{"chunk_id": "chunk-reference"}]},
            },
        },
        {
            "role": "tool",
            "tool_call_id": "queue-full",
            "name": "queue_status_query",
            "content": {
                "tool_name": "queue_status_query",
                "call_id": "queue-full",
                "ok": True,
                "completed_at": "2026-07-16T09:08:00+08:00",
                "data": {"appointment_id": 1, "queue_status": {"waiting_count": 4}},
            },
        },
    ]

    projected = project_historical_messages(
        [
            {"role": "user", "content": "上一轮"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "knowledge-full", "name": "knowledge_search", "args": {}},
                    {"id": "knowledge-reference", "name": "knowledge_search", "args": {}},
                    {"id": "queue-full", "name": "queue_status_query", "args": {}},
                ],
            },
            *messages,
        ]
    )
    knowledge, business = recent_tool_context(projected)

    assert [item["chunk_id"] for item in knowledge] == ["chunk-full"]
    assert knowledge[0]["evidence_text"] == "应当交验身份证明原件。"
    assert [item["call_id"] for item in business] == ["queue-full"]
    assert business[0]["data"]["queue_status"]["waiting_count"] == 4


def test_business_citation_cannot_support_a_different_application_id():
    decision = validate_grounded_answer(
        {
            "query": "我的申请状态",
            "retrieved_sources": [],
            "tool_results": [
                {
                    "tool_name": "ticket_detail_query",
                    "call_id": "detail-15",
                    "ok": True,
                    "data": {"applications": [{"application_id": 15, "status": "accepted"}]},
                }
            ],
        },
        "application_id: 99 当前状态为已受理。[业务1]",
    )

    assert decision["ok"] is False
    assert decision["reason"] == "business_key_mismatch"


def test_historical_business_snapshot_requires_time_context():
    state = {
        "query": "刚才排队情况是什么",
        "retrieved_sources": [],
        "tool_results": [
            {
                "tool_name": "queue_status_query",
                "call_id": "queue-old",
                "ok": True,
                "memory_origin": "historical_snapshot",
                "observed_at": "2026-07-16T09:08:00+08:00",
                "data": {"appointment_id": 4101, "queue_status": {"waiting_count": 4}},
            }
        ],
    }

    missing_time = validate_grounded_answer(state, "前方等待人数为4人。[业务1]")
    with_time = validate_grounded_answer(state, "截至刚才查询时，前方等待人数为4人。[业务1]")

    assert missing_time["ok"] is False
    assert missing_time["reason"] == "historical_snapshot_without_time_context"
    assert with_time["ok"] is True
