from pathlib import Path
import sys
import types

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.rag.model_providers.embeddings.qwen3_vl_embedding_provider import Qwen3VLEmbeddingProvider


class FakeDashScopeResponse:
    status_code = 200

    def __init__(self, output):
        self.output = output


class FakeMultiModalEmbeddingItemText:
    def __init__(self, *, text, factor):
        self.text = text
        self.factor = factor


class FakeMultiModalEmbeddingItemImage:
    def __init__(self, *, image, factor):
        self.image = image
        self.factor = factor


class FakeMultiModalEmbedding:
    @staticmethod
    def call(**kwargs):
        raise AssertionError("fake_call should be patched by each test")


def install_fake_dashscope(monkeypatch):
    fake_module = types.ModuleType("dashscope")
    fake_module.MultiModalEmbedding = FakeMultiModalEmbedding
    fake_module.MultiModalEmbeddingItemText = FakeMultiModalEmbeddingItemText
    fake_module.MultiModalEmbeddingItemImage = FakeMultiModalEmbeddingItemImage
    monkeypatch.setitem(sys.modules, "dashscope", fake_module)
    return fake_module


def test_qwen3_vl_embedding_batches_text(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return FakeDashScopeResponse(
            {
                "embeddings": [
                    {"embedding": [1.0, 0.0, 0.0]},
                    {"embedding": [0.0, 1.0, 0.0]},
                ]
            }
        )

    dashscope = install_fake_dashscope(monkeypatch)
    monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", fake_call)
    provider = Qwen3VLEmbeddingProvider(api_key="sk-test", dimension=96)

    vectors = provider.embed_documents(["办理材料", "办理流程"])

    assert provider.dimension == 1024
    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert calls[0]["model"] == "qwen3-vl-embedding"
    assert calls[0]["dimension"] == 1024
    assert calls[0].get("enable_fusion") is not True
    assert [item.text for item in calls[0]["input"]] == ["办理材料", "办理流程"]


def test_qwen3_vl_embedding_splits_large_text_batches(monkeypatch):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return FakeDashScopeResponse(
            {
                "embeddings": [
                    {"embedding": [float(index), 1.0]}
                    for index, _ in enumerate(kwargs["input"])
                ]
            }
        )

    dashscope = install_fake_dashscope(monkeypatch)
    monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", fake_call)
    provider = Qwen3VLEmbeddingProvider(api_key="sk-test", dimension=1024)

    vectors = provider.embed_documents([f"文本{index}" for index in range(45)])

    assert len(vectors) == 45
    assert [len(call["input"]) for call in calls] == [20, 20, 5]
    assert calls[0]["input"][0].text == "文本0"
    assert calls[1]["input"][0].text == "文本20"
    assert calls[2]["input"][0].text == "文本40"


def test_qwen3_vl_embedding_uses_image_asset_for_image_chunk(monkeypatch, tmp_path):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        if kwargs.get("enable_fusion"):
            return FakeDashScopeResponse({"embedding": [0.2, 0.8]})
        return FakeDashScopeResponse({"embeddings": [{"embedding": [0.9, 0.1]}]})

    dashscope = install_fake_dashscope(monkeypatch)
    monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", fake_call)
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"fake png")
    provider = Qwen3VLEmbeddingProvider(api_key="sk-test", dimension=512, image_res_level=2)
    chunks = [
        KnowledgeChunk(
            id="c1",
            doc_id="d1",
            parent_id="",
            chunk_type="child",
            title="文本",
            content="普通文本证据",
        ),
        KnowledgeChunk(
            id="c2",
            doc_id="d1",
            parent_id="",
            chunk_type="child",
            title="图片",
            content="PDF 第 1 页图片证据",
            metadata={
                "modality": "image",
                "asset_path": str(image_path),
                "asset_text": "图片上的申请表文字",
            },
        ),
    ]

    vectors = provider.embed_chunks(chunks)

    assert vectors == [[0.9, 0.1], [0.2, 0.8]]
    assert len(calls) == 2
    image_call = calls[1]
    assert image_call["enable_fusion"] is True
    assert image_call["res_level"] == 2
    assert image_call["input"][0].image == str(image_path)
    assert image_call["input"][1].text == "图片上的申请表文字"


def test_qwen3_vl_embedding_prefers_compact_image_embedding_text(monkeypatch, tmp_path):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return FakeDashScopeResponse({"embedding": [0.2, 0.8]})

    dashscope = install_fake_dashscope(monkeypatch)
    monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", fake_call)
    image_path = tmp_path / "picture.png"
    image_path.write_bytes(b"fake png")
    provider = Qwen3VLEmbeddingProvider(api_key="sk-test", dimension=512)
    chunk = KnowledgeChunk(
        id="c-image",
        doc_id="d1",
        parent_id="",
        chunk_type="child",
        title="图片",
        content="图片证据",
        metadata={
            "modality": "image",
            "asset_path": str(image_path),
            "asset_text": "完整 OCR" * 2000,
            "image_embedding_text": "视觉摘要\nOCR预览",
        },
    )

    provider.embed_chunks([chunk])

    assert calls[0]["input"][1].text == "视觉摘要\nOCR预览"


def test_qwen3_vl_embedding_uses_fusion_for_image_query_with_text(monkeypatch, tmp_path):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return FakeDashScopeResponse({"embedding": [0.3, 0.7]})

    dashscope = install_fake_dashscope(monkeypatch)
    monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", fake_call)
    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"fake png")
    provider = Qwen3VLEmbeddingProvider(api_key="sk-test", dimension=512)

    vector = provider.embed_image_query(str(image_path), text="这张材料缺什么")

    assert vector == [0.3, 0.7]
    assert calls[0]["enable_fusion"] is True
    assert calls[0]["input"][0].image == str(image_path)
    assert calls[0]["input"][1].text == "这张材料缺什么"


def test_qwen3_vl_embedding_uses_image_only_for_image_query_without_text(monkeypatch, tmp_path):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return FakeDashScopeResponse({"embeddings": [{"embedding": [0.4, 0.6]}]})

    dashscope = install_fake_dashscope(monkeypatch)
    monkeypatch.setattr(dashscope.MultiModalEmbedding, "call", fake_call)
    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"fake png")
    provider = Qwen3VLEmbeddingProvider(api_key="sk-test", dimension=512)

    vector = provider.embed_image_query(str(image_path), text="")

    assert vector == [0.4, 0.6]
    assert calls[0].get("enable_fusion") is not True
    assert len(calls[0]["input"]) == 1
    assert calls[0]["input"][0].image == str(image_path)
