from declaration_qa.rag.context_engineering import build_short_term_context, normalize_recent_messages


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


def test_build_short_term_context_summarizes_ticket_image_and_evidence():
    context = build_short_term_context(
        {
            "query": "这个材料怎么补",
            "normalized_query": "这个材料怎么补",
            "intent": "correction",
            "route": "ticket_context_related",
            "session_messages": [{"role": "user", "content": "上一轮问了办理进度"}],
            "query_image": {"filename": "flow.png", "content_type": "image/png", "size_bytes": 1234, "text_present": True},
            "apply_context": {"apply_id": "A001", "biz_type": "water", "title": "取水许可", "status": "退回"},
            "reject_info": {"reject_reason": "申请书缺少签字", "material_name": "取水许可申请书"},
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
    assert context["metrics"]["has_ticket_context"] is True
    assert context["metrics"]["has_query_image"] is True
    assert context["ticket_context"]["apply_id"] == "A001"
    assert context["ticket_context"]["reject_reason"] == "申请书缺少签字"
    assert context["query_image"]["filename"] == "flow.png"
    assert context["evidence_overview"][0]["chunk_id"] == "chunk-1"
