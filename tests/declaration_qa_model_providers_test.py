from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.core.domain import KnowledgeChunk, RetrievedSource
from declaration_qa.rag.model_providers.factories import build_reranker
from declaration_qa.rag.model_providers.rerankers import DashScopeReranker
from declaration_qa.rag.model_providers.streaming import stream_chat_model_content


def test_stream_chat_model_content_yields_langchain_chunk_text():
    chunks = [
        SimpleNamespace(content="申报"),
        SimpleNamespace(content=[{"type": "text", "text": "进度"}, "查询"]),
        "",
        SimpleNamespace(content=None),
    ]

    assert list(stream_chat_model_content(chunks)) == ["申报", "进度", "查询"]


def test_dashscope_reranker_uses_sdk_text_rerank(monkeypatch):
    captured = {}

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status_code=HTTPStatus.OK,
                output={
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.42},
                    ]
                },
            )

    fake_dashscope = SimpleNamespace(TextReRank=FakeTextReRank, api_key="")
    monkeypatch.setitem(__import__("sys").modules, "dashscope", fake_dashscope)
    sources = [
        RetrievedSource(
            chunk=KnowledgeChunk(
                id="chunk-a",
                doc_id="doc",
                parent_id="",
                chunk_type="child",
                title="A",
                content="办理时限为10个工作日。",
            ),
            score=0.2,
        ),
        RetrievedSource(
            chunk=KnowledgeChunk(
                id="chunk-b",
                doc_id="doc",
                parent_id="",
                chunk_type="child",
                title="B",
                content="承诺办结时限为5个工作日。",
            ),
            score=0.3,
        ),
    ]

    reranked = DashScopeReranker(api_key="sdk-key", model="qwen3-rerank").rerank("多久办完", sources, {}, top_n=2)

    assert fake_dashscope.api_key == "sdk-key"
    assert captured["model"] == "qwen3-rerank"
    assert captured["query"] == "多久办完"
    assert captured["documents"] == ["办理时限为10个工作日。", "承诺办结时限为5个工作日。"]
    assert captured["top_n"] == 2
    assert captured["instruct"]
    assert [source.chunk.id for source in reranked] == ["chunk-b", "chunk-a"]
    assert reranked[0].rerank_score == 0.91


def test_build_reranker_uses_dashscope_sdk_provider():
    reranker = build_reranker(
        DeclarationQASettings(
            _env_file=None,
            DECLARATION_QA_RERANK_PROVIDER="dashscope",
            DECLARATION_QA_RERANK_API_KEY="sdk-key",
            DECLARATION_QA_RERANK_MODEL="qwen3-rerank",
        )
    )

    assert isinstance(reranker, DashScopeReranker)
    assert reranker.model == "qwen3-rerank"
