"""DashScope SDK 重排器。"""

from __future__ import annotations

import base64
import mimetypes
from http import HTTPStatus
from pathlib import Path
from typing import Any

from declaration_qa.core.domain import RetrievedSource
from declaration_qa.observability.performance import performance_stage

QWEN3_VL_RERANK_MODELS = {"qwen3-vl-rerank"}
MAX_VL_DOCUMENTS = 100
MAX_VL_IMAGE_DOCUMENTS = 40
MAX_RERANK_TEXT_CHARS = 8000


class DashScopeReranker:
    """DashScope SDK 重排器适配器。"""

    name = "dashscope"
    default_instruct = "根据用户问题，从候选材料中找出最相关、最能支撑回答的证据。"

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "",
        text_model: str = "",
        instruct: str = "",
    ) -> None:
        """初始化DashScopeReranker。"""
        self.api_key = api_key
        self.model = model or "qwen3-vl-rerank"
        self.text_model = text_model or "qwen3-rerank"
        self.instruct = instruct or self.default_instruct

    def rerank(
        self,
        query: str,
        sources: list[Any],
        filters: dict[str, Any],
        *,
        top_n: int,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> list[Any]:
        """按查询相关性重新排序候选来源。"""
        if not sources:
            return []
        if self.is_vl_model and rerank_query_is_multimodal(retrieval_plan):
            return self._rerank_multimodal(query, sources, top_n=top_n, retrieval_plan=retrieval_plan)
        return self._rerank_text(query, sources, top_n=top_n)

    @property
    def is_vl_model(self) -> bool:
        """判断当前模型是否支持多模态重排。"""
        return self.model.strip().lower() in QWEN3_VL_RERANK_MODELS

    def _rerank_text(self, query: str, sources: list[Any], *, top_n: int) -> list[Any]:
        """调用 DashScope 文本重排。"""
        dashscope = import_dashscope()
        with performance_stage(
            "dashscope.rerank",
            provider=self.name,
            model=self.text_model,
            candidate_count=len(sources),
            top_n=max(1, int(top_n)),
            multimodal=False,
        ):
            response = dashscope.TextReRank.call(
                model=self.text_model,
                query=query,
                documents=[source.chunk.content for source in sources],
                top_n=max(1, int(top_n)),
                api_key=self.api_key or None,
                instruct=self.instruct,
            )
        if response_value(response, "status_code") != HTTPStatus.OK:
            code = response_value(response, "code", "")
            message = response_value(response, "message", "")
            raise RuntimeError(f"DashScope rerank failed: {code} {message}".strip())
        output = response_value(response, "output", {}) or {}
        results = response_value(output, "results", []) or response_value(response, "results", [])
        reranked = []
        for item in results or []:
            index = int(response_value(item, "index", -1))
            score = float(response_value(item, "score", None) or response_value(item, "relevance_score", 0.0) or 0.0)
            if index < 0 or index >= len(sources):
                continue
            source = sources[index]
            reranked.append(clone_with_rerank_score(source, score))
        return sorted(reranked, key=lambda item: (item.rerank_score, item.score), reverse=True)[:top_n]
    def _rerank_multimodal(
        self,
        query: str,
        sources: list[Any],
        *,
        top_n: int,
        retrieval_plan: dict[str, Any] | None,
    ) -> list[Any]:
        """调用 DashScope Qwen3-VL 多模态重排。"""
        dashscope = import_dashscope()
        documents, source_indexes = build_vl_documents(sources)
        if not documents:
            return []
        with performance_stage(
            "dashscope.rerank",
            provider=self.name,
            model=self.model,
            candidate_count=len(documents),
            top_n=max(1, int(top_n)),
            multimodal=True,
        ):
            response = dashscope.TextReRank.call(
                model=self.model,
                query=build_vl_query(query, retrieval_plan),
                documents=documents,
                top_n=max(1, int(top_n)),
                api_key=self.api_key or None,
                instruct=self.instruct,
                return_documents=False,
            )
        if response_value(response, "status_code") != HTTPStatus.OK:
            code = response_value(response, "code", "")
            message = response_value(response, "message", "")
            raise RuntimeError(f"DashScope rerank failed: {code} {message}".strip())
        output = response_value(response, "output", {}) or {}
        results = response_value(output, "results", []) or response_value(response, "results", [])
        reranked = []
        for item in results or []:
            document_index = int(response_value(item, "index", -1))
            score = float(response_value(item, "score", None) or response_value(item, "relevance_score", 0.0) or 0.0)
            if document_index < 0 or document_index >= len(source_indexes):
                continue
            source = sources[source_indexes[document_index]]
            reranked.append(clone_with_rerank_score(source, score))
        return sorted(reranked, key=lambda item: (item.rerank_score, item.score), reverse=True)[:top_n]


def rerank_query_is_multimodal(retrieval_plan: dict[str, Any] | None) -> bool:
    """仅允许图片查询向多模态重排模型发送图片载荷。"""
    plan = retrieval_plan or {}
    return str(plan.get("query_modality") or "text").strip().lower() == "image"


def response_value(payload: Any, key: str, default: Any = None) -> Any:
    """从 DashScope SDK 响应对象或字典读取值。"""
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def build_vl_query(query: str, retrieval_plan: dict[str, Any] | None) -> dict[str, str]:
    """构建 Qwen3-VL-Rerank 查询输入。"""
    plan = retrieval_plan or {}
    query_modality = str(plan.get("query_modality") or "").strip().lower()
    image_path = str(plan.get("query_image_path") or "").strip()
    if query_modality == "image" and image_path:
        image_payload = image_data_uri(image_path)
        if image_payload:
            return {"image": image_payload}
    return {"text": safe_rerank_text(query)}


def build_vl_documents(sources: list[Any]) -> tuple[list[dict[str, str]], list[int]]:
    """把候选来源转换为 Qwen3-VL-Rerank 支持的文档列表。"""
    documents: list[dict[str, str]] = []
    source_indexes: list[int] = []
    image_document_count = 0
    for index, source in enumerate(sources):
        if len(documents) >= MAX_VL_DOCUMENTS:
            break
        document: dict[str, str] | None = None
        if source_is_image(source) and image_document_count < MAX_VL_IMAGE_DOCUMENTS:
            image_payload = image_data_uri(str(source.chunk.metadata.get("asset_path") or ""))
            if image_payload:
                document = {"image": image_payload}
                image_document_count += 1
        if document is None:
            text = safe_rerank_text(getattr(source.chunk, "content", ""))
            if not text:
                continue
            document = {"text": text}
        documents.append(document)
        source_indexes.append(index)
    return documents, source_indexes


def source_is_image(source: Any) -> bool:
    """判断候选来源是否有可用于多模态重排的图片资产。"""
    metadata = getattr(getattr(source, "chunk", None), "metadata", {}) or {}
    return metadata.get("modality") == "image" and bool(metadata.get("asset_path"))


def image_data_uri(path: str) -> str:
    """将本地图片文件转换为可发送给 DashScope 的 Data URI。"""
    image_path = Path(path)
    if not image_path.is_file():
        return ""
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def safe_rerank_text(text: str) -> str:
    """返回非空且长度受控的重排文本。"""
    normalized = str(text or "").strip()
    return (normalized or "图片内容相关申报材料")[:MAX_RERANK_TEXT_CHARS]


def clone_with_rerank_score(source: Any, score: float) -> RetrievedSource:
    """复制来源并写入重排分数。"""
    return RetrievedSource(
        chunk=source.chunk,
        score=max(source.score, score),
        dense_score=source.dense_score,
        bm25_score=source.bm25_score,
        rerank_score=score,
        retrieval_channel=f"{source.retrieval_channel}+rerank",
    )


def import_dashscope():
    """延迟导入 DashScope SDK。"""
    import dashscope

    return dashscope
