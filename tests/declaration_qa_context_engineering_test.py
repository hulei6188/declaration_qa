from declaration_qa.memory.context import build_short_term_context


def test_build_short_term_context_summarizes_current_tool_image_and_evidence_without_binding():
    context = build_short_term_context(
        {
            "query": "这个材料怎么补",
            "normalized_query": "这个材料怎么补",
            "intent": "correction",
            "route": "agentic_rag",
            "agent_messages": [
                {"role": "user", "content": "上一轮问了办理进度"},
                {"role": "assistant", "content": "请提供申请编号"},
            ],
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

    assert context["metrics"]["agent_message_count"] == 2
    assert context["metrics"]["has_business_tool_context"] is True
    assert context["metrics"]["has_query_image"] is True
    assert context["business_tool_context"]["item_code"] == "WATER-001"
    assert context["business_tool_context"]["selected_application_id"] == 101
    assert context["business_tool_context"]["application_count"] == 2
    assert context["query_image"]["filename"] == "flow.png"
    assert context["evidence_overview"][0]["chunk_id"] == "chunk-1"
