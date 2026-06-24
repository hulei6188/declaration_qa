from __future__ import annotations

from types import SimpleNamespace

from declaration_qa.rag.model_providers.streaming import stream_chat_model_content


def test_stream_chat_model_content_yields_langchain_chunk_text():
    chunks = [
        SimpleNamespace(content="申报"),
        SimpleNamespace(content=[{"type": "text", "text": "进度"}, "查询"]),
        "",
        SimpleNamespace(content=None),
    ]

    assert list(stream_chat_model_content(chunks)) == ["申报", "进度", "查询"]
