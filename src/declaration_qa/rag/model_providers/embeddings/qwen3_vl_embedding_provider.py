"""DashScope Qwen3-VL 多模态向量化 provider。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from declaration_qa.core.domain import KnowledgeChunk


QWEN3_VL_SUPPORTED_DIMENSIONS = {256, 512, 768, 1024, 1536, 2048, 2560}


class Qwen3VLEmbeddingProvider:
    """基于 qwen3-vl-embedding 的文本/图片统一向量化实现。"""

    name = "dashscope-qwen3-vl"
    document_batch_size = 10

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-vl-embedding",
        dimension: int = 1024,
        image_res_level: int = 0,
    ) -> None:
        """初始化 Qwen3VLEmbeddingProvider。"""
        if not api_key:
            raise ValueError("Qwen3-VL embedding provider requires an API key")
        self.api_key = api_key
        self.model = model or "qwen3-vl-embedding"
        self.dimension = normalize_qwen3_vl_dimension(dimension)
        self.image_res_level = int(image_res_level or 0)

    def embed_query(self, text: str) -> list[float]:
        """为单条查询文本生成向量表示。"""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为多条文档文本生成向量表示。"""
        if not texts:
            return []
        from dashscope import MultiModalEmbeddingItemText

        items = [MultiModalEmbeddingItemText(text=safe_embedding_text(text), factor=1.0) for text in texts]
        return self._call(items, enable_fusion=False)

    def embed_chunks(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        """根据 chunk 模态生成文本向量或图片向量。"""
        vectors: list[list[float] | None] = [None] * len(chunks)
        pending_texts: list[str] = []
        pending_indices: list[int] = []
        for index, chunk in enumerate(chunks):
            if chunk_is_image(chunk):
                flush_text_embeddings(self, pending_texts, pending_indices, vectors)
                pending_texts = []
                pending_indices = []
                vectors[index] = self.embed_image_chunk(chunk)
                continue
            pending_texts.append(chunk.content)
            pending_indices.append(index)
        flush_text_embeddings(self, pending_texts, pending_indices, vectors)
        return [vector or zero_vector(self.dimension) for vector in vectors]

    def embed_image_chunk(self, chunk: KnowledgeChunk) -> list[float]:
        """为图片 chunk 生成图片或图文融合向量。"""
        from dashscope import MultiModalEmbeddingItemImage, MultiModalEmbeddingItemText

        asset_path = str(chunk.metadata.get("asset_path") or "").strip()
        if not asset_path or not Path(asset_path).is_file():
            return self.embed_query(chunk.content)
        items: list[Any] = [MultiModalEmbeddingItemImage(image=asset_path, factor=1.0)]
        text = image_chunk_text(chunk)
        if text:
            items.append(MultiModalEmbeddingItemText(text=safe_embedding_text(text), factor=0.25))
        return self._call(items, enable_fusion=bool(text))[0]

    def _call(self, items: list[Any], *, enable_fusion: bool) -> list[list[float]]:
        """调用 DashScope MultiModalEmbedding 并解析向量结果。"""
        from dashscope import MultiModalEmbedding

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": items,
            "api_key": self.api_key,
            "dimension": self.dimension,
            "output_type": "dense",
            "auto_truncation": True,
        }
        if enable_fusion:
            kwargs["enable_fusion"] = True
        if self.image_res_level > 0:
            kwargs["res_level"] = self.image_res_level
        response = MultiModalEmbedding.call(**kwargs)
        return parse_dashscope_embedding_response(response, expected_count=1 if enable_fusion else len(items))


def flush_text_embeddings(
    provider: Qwen3VLEmbeddingProvider,
    texts: list[str],
    indices: list[int],
    vectors: list[list[float] | None],
) -> None:
    """将累积的文本 chunk 批量向量化并写回原位置。"""
    if not texts:
        return
    embedded = provider.embed_documents(texts)
    for index, vector in zip(indices, embedded):
        vectors[index] = vector


def chunk_is_image(chunk: KnowledgeChunk) -> bool:
    """判断 chunk 是否代表视觉资产。"""
    metadata = chunk.metadata or {}
    return metadata.get("modality") == "image" and bool(metadata.get("asset_path"))


def image_chunk_text(chunk: KnowledgeChunk) -> str:
    """返回图片向量融合时使用的辅助文本。"""
    metadata = chunk.metadata or {}
    return str(metadata.get("asset_text") or chunk.content or "").strip()


def safe_embedding_text(text: str) -> str:
    """返回非空、长度受控的向量化文本。"""
    normalized = str(text or "").strip()
    if not normalized:
        return "空内容"
    return normalized[:6000]


def normalize_qwen3_vl_dimension(dimension: int) -> int:
    """归一化 qwen3-vl-embedding 支持的向量维度。"""
    value = int(dimension or 0)
    if value in QWEN3_VL_SUPPORTED_DIMENSIONS:
        return value
    return 1024


def parse_dashscope_embedding_response(response: Any, *, expected_count: int) -> list[list[float]]:
    """从 DashScope 响应对象中提取向量列表。"""
    status_code = int(getattr(response, "status_code", 200) or 200)
    if status_code >= 400:
        message = getattr(response, "message", "") or getattr(response, "code", "") or str(response)
        raise RuntimeError(f"DashScope embedding failed: HTTP {status_code} {message}")
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not isinstance(output, dict):
        output = {}
    vectors = vectors_from_output(output)
    if len(vectors) < expected_count:
        raise RuntimeError("DashScope embedding response does not contain enough vectors")
    return vectors[:expected_count]


def vectors_from_output(output: dict[str, Any]) -> list[list[float]]:
    """兼容 DashScope 不同版本返回结构提取向量。"""
    if isinstance(output.get("embedding"), list):
        return [normalize_vector(output["embedding"])]
    candidates = output.get("embeddings") or output.get("vectors") or output.get("data") or []
    vectors: list[list[float]] = []
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict):
                raw = item.get("embedding") or item.get("vector") or item.get("dense_embedding")
            else:
                raw = item
            if isinstance(raw, list):
                vectors.append(normalize_vector(raw))
    return vectors


def normalize_vector(vector: list[Any]) -> list[float]:
    """将向量值转换为 float。"""
    return [float(value) for value in vector]


def zero_vector(dimension: int) -> list[float]:
    """构建兜底零向量。"""
    return [0.0] * int(dimension or 1)
