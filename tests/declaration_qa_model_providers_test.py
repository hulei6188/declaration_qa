from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.core.domain import KnowledgeChunk, RetrievedSource
from declaration_qa.rag.model_providers.factories import build_agent_model, build_reranker
from declaration_qa.rag.model_providers.rerankers.dashscope_reranker import DashScopeReranker, safe_rerank_text
from declaration_qa.rag.model_providers.streaming import stream_chat_model_content


def test_stream_chat_model_content_yields_langchain_chunk_text():
    chunks = [
        SimpleNamespace(content="申报"),
        SimpleNamespace(content=[{"type": "text", "text": "进度"}, "查询"]),
        "",
        SimpleNamespace(content=None),
    ]

    assert list(stream_chat_model_content(chunks)) == ["申报", "进度", "查询"]


def test_build_agent_model_uses_independent_deepseek_connection(monkeypatch):
    captured = {}
    model = object()

    def fake_create_chat_openai(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(
        "declaration_qa.rag.model_providers.factories.create_chat_openai",
        fake_create_chat_openai,
    )
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_LLM_PROVIDER="deepseek",
        DECLARATION_QA_OPENAI_API_KEY="dashscope-key",
        DECLARATION_QA_OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    result = build_agent_model(
        settings,
        model_name="deepseek-v4-flash",
        api_key="deepseek-key",
        base_url="https://api.deepseek.com",
    )

    assert result is model
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["api_key"] == "deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.com"


def test_empty_text_key_does_not_fall_back_to_dashscope_key():
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_LLM_PROVIDER="deepseek",
        DECLARATION_QA_OPENAI_API_KEY="dashscope-key",
    )

    assert build_agent_model(
        settings,
        model_name="deepseek-v4-flash",
        api_key="",
        base_url="https://api.deepseek.com",
    ) is None


def test_dashscope_vl_rerank_text_respects_document_limit():
    assert len(safe_rerank_text("材" * 9000)) == 8000


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

    assert fake_dashscope.api_key == ""
    assert captured["api_key"] == "sdk-key"
    assert captured["model"] == "qwen3-rerank"
    assert captured["query"] == "多久办完"
    assert captured["documents"] == ["办理时限为10个工作日。", "承诺办结时限为5个工作日。"]
    assert captured["top_n"] == 2
    assert captured["instruct"]
    assert [source.chunk.id for source in reranked] == ["chunk-b", "chunk-a"]
    assert reranked[0].rerank_score == 0.91


def test_dashscope_vl_reranker_uses_text_model_for_text_query_without_dropping_candidates(monkeypatch):
    captured = {}

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status_code=HTTPStatus.OK,
                output={"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}]},
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "dashscope",
        SimpleNamespace(TextReRank=FakeTextReRank, api_key=""),
    )
    sources = [
        RetrievedSource(
            chunk=KnowledgeChunk(
                id="text-chunk",
                doc_id="doc",
                parent_id="",
                chunk_type="child",
                title="文本",
                content="涉外收养登记材料清单。",
            ),
            score=0.2,
        ),
        RetrievedSource(
            chunk=KnowledgeChunk(
                id="image-chunk",
                doc_id="doc",
                parent_id="",
                chunk_type="child",
                title="图片 OCR",
                content="授权委托书应当签字并认证。",
                metadata={"modality": "image", "asset_path": "unused-for-text-query.png"},
            ),
            score=0.3,
        ),
    ]

    reranked = DashScopeReranker(
        api_key="sdk-key",
        model="qwen3-vl-rerank",
        text_model="qwen3-rerank",
    ).rerank(
        "涉外收养登记需要哪些材料",
        sources,
        {},
        top_n=2,
        retrieval_plan={"query_modality": "text"},
    )

    assert captured["model"] == "qwen3-rerank"
    assert captured["query"] == "涉外收养登记需要哪些材料"
    assert captured["documents"] == [source.chunk.content for source in sources]
    assert len(captured["documents"]) == len(sources) == 2
    assert "return_documents" not in captured
    assert [source.chunk.id for source in reranked] == ["image-chunk", "text-chunk"]


def test_dashscope_vl_reranker_uses_image_query_and_image_documents(monkeypatch, tmp_path):
    captured = {}

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                status_code=HTTPStatus.OK,
                output={
                    "results": [
                        {"index": 1, "relevance_score": 0.93},
                        {"index": 0, "relevance_score": 0.51},
                    ]
                },
            )

    fake_dashscope = SimpleNamespace(TextReRank=FakeTextReRank, api_key="")
    monkeypatch.setitem(__import__("sys").modules, "dashscope", fake_dashscope)
    query_image = tmp_path / "query.png"
    query_image.write_bytes(b"query image bytes")
    chunk_image = tmp_path / "chunk.png"
    chunk_image.write_bytes(b"chunk image bytes")
    sources = [
        RetrievedSource(
            chunk=KnowledgeChunk(
                id="text-chunk",
                doc_id="doc",
                parent_id="",
                chunk_type="child",
                title="文本",
                content="取水许可申请流程包括受理、审查、决定。",
            ),
            score=0.2,
        ),
        RetrievedSource(
            chunk=KnowledgeChunk(
                id="image-chunk",
                doc_id="doc",
                parent_id="",
                chunk_type="child",
                title="流程图",
                content="取水许可流程图 OCR 文本",
                metadata={"modality": "image", "asset_path": str(chunk_image)},
            ),
            score=0.3,
        ),
    ]

    reranked = DashScopeReranker(api_key="sdk-key", model="qwen3-vl-rerank").rerank(
        "请根据这张图片检索相关申报材料",
        sources,
        {},
        top_n=2,
        retrieval_plan={"query_modality": "image", "query_image_path": str(query_image)},
    )

    assert captured["model"] == "qwen3-vl-rerank"
    assert captured["query"]["image"].startswith("data:image/png;base64,")
    assert captured["documents"][0] == {"text": "取水许可申请流程包括受理、审查、决定。"}
    assert captured["documents"][1]["image"].startswith("data:image/png;base64,")
    assert captured["api_key"] == "sdk-key"
    assert captured["return_documents"] is False
    assert [source.chunk.id for source in reranked] == ["image-chunk", "text-chunk"]
    assert reranked[0].rerank_score == 0.93


def test_build_reranker_uses_dashscope_sdk_provider():
    reranker = build_reranker(
        DeclarationQASettings(
            _env_file=None,
            DECLARATION_QA_RERANK_PROVIDER="dashscope",
            DECLARATION_QA_RERANK_API_KEY="sdk-key",
            DECLARATION_QA_RERANK_MODEL="qwen3-vl-rerank",
            DECLARATION_QA_TEXT_RERANK_MODEL="qwen3-rerank",
        )
    )

    assert isinstance(reranker, DashScopeReranker)
    assert reranker.model == "qwen3-vl-rerank"
    assert reranker.text_model == "qwen3-rerank"
