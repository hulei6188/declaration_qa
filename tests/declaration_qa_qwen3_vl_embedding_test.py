from pathlib import Path

from declaration_qa.core.domain import KnowledgeChunk
from declaration_qa.rag.model_providers.embeddings.qwen3_vl_embedding_provider import Qwen3VLEmbeddingProvider


class FakeDashScopeResponse:
    status_code = 200

    def __init__(self, output):
        self.output = output


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

    monkeypatch.setattr("dashscope.MultiModalEmbedding.call", fake_call)
    provider = Qwen3VLEmbeddingProvider(api_key="sk-test", dimension=96)

    vectors = provider.embed_documents(["办理材料", "办理流程"])

    assert provider.dimension == 1024
    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert calls[0]["model"] == "qwen3-vl-embedding"
    assert calls[0]["dimension"] == 1024
    assert calls[0].get("enable_fusion") is not True
    assert [item.text for item in calls[0]["input"]] == ["办理材料", "办理流程"]


def test_qwen3_vl_embedding_uses_image_asset_for_image_chunk(monkeypatch, tmp_path):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        if kwargs.get("enable_fusion"):
            return FakeDashScopeResponse({"embedding": [0.2, 0.8]})
        return FakeDashScopeResponse({"embeddings": [{"embedding": [0.9, 0.1]}]})

    monkeypatch.setattr("dashscope.MultiModalEmbedding.call", fake_call)
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
