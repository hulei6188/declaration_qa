from declaration_qa.memory.context import build_short_term_context, normalize_recent_messages


def test_normalize_recent_messages_keeps_last_five_turns_and_truncates():
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"消息{index}" + "很长" * 300}
        for index in range(12)
    ]

    recent = normalize_recent_messages(messages)

    assert len(recent) == 10
    assert recent[0]["content"].startswith("消息2")
    assert all(not item["content"].startswith("消息0") for item in recent)
    assert all(len(item["content"]) <= 503 for item in recent)


def test_build_short_term_context_summarizes_current_tool_image_and_evidence_without_binding():
    context = build_short_term_context(
        {
            "query": "这个材料怎么补",
            "normalized_query": "这个材料怎么补",
            "intent": "correction",
            "route": "agentic_rag",
            "session_messages": [{"role": "user", "content": "上一轮问了办理进度"}],
            "query_image": {"filename": "flow.png", "content_type": "image/png", "size_bytes": 1234, "text_present": True},
            "business_snapshot": {
                "source": "ticket_detail_query",
                "item_code": "WATER-001",
                "selected_application_id": 101,
                "applications": [{"application_id": 101}, {"application_id": 88}],
            },
            "retrieved_sources": [
                {
                    "chunk_id": "chunk-1",
                    "title": "item_28.md",
                    "score": 0.9,
                    "rerank_score": 0.88,
                    "metadata": {
                        "chunk_type": "attachment_text",
                        "section_path": "取水许可办事指南 > 申请材料",
                        "node_title": "取水许可申请书",
                    },
                }
            ],
        }
    )

    assert context["metrics"]["recent_message_count"] == 1
    assert context["metrics"]["has_business_tool_context"] is True
    assert context["metrics"]["has_query_image"] is True
    assert context["business_tool_context"]["item_code"] == "WATER-001"
    assert context["business_tool_context"]["selected_application_id"] == 101
    assert context["business_tool_context"]["application_count"] == 2
    assert context["query_image"]["filename"] == "flow.png"
    assert context["evidence_overview"][0]["chunk_id"] == "chunk-1"
