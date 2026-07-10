"""Langfuse 可观测性辅助函数。"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import Any

from declaration_qa.core.config import DeclarationQASettings


SENSITIVE_KEYWORDS = (
    "id_card",
    "idcard",
    "identity",
    "phone",
    "mobile",
    "secret",
    "token",
    "password",
    "api_key",
    "authorization",
)
MAX_STRING_LENGTH = 800
MAX_LIST_ITEMS = 12
MAX_DICT_ITEMS = 30
CURRENT_TRACE_ID: ContextVar[str] = ContextVar("declaration_qa_langfuse_trace_id", default="")
CURRENT_TRACE_DEPTH: ContextVar[int] = ContextVar("declaration_qa_langfuse_trace_depth", default=0)
CURRENT_SPAN_ID: ContextVar[str] = ContextVar("declaration_qa_langfuse_span_id", default="")


class NoopObservation:
    """空实现对象，提供本应用使用的 Langfuse 观测方法。"""

    trace_id = ""

    def update(self, **_: Any) -> None:
        """可观测性关闭时忽略更新。"""

    def set_trace_io(self, **_: Any) -> None:
        """可观测性关闭时忽略 trace 输入输出更新。"""

    def score_trace(self, **_: Any) -> None:
        """可观测性关闭时忽略 trace 评分。"""

    def record_error(self, *_: Any, **__: Any) -> None:
        """可观测性关闭时忽略错误记录。"""


def langfuse_ready(settings: DeclarationQASettings | None = None) -> bool:
    """返回是否应尝试 Langfuse 追踪。"""
    settings = ensure_langfuse_env(settings)
    if settings is not None:
        return bool(settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key)
    if os.getenv("DECLARATION_QA_LANGFUSE_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


@lru_cache
def default_settings() -> DeclarationQASettings | None:
    """从环境变量和 .env 加载可观测性辅助函数所需的项目配置。"""
    try:
        return DeclarationQASettings()
    except Exception:
        return None


def ensure_langfuse_env(settings: DeclarationQASettings | None = None) -> DeclarationQASettings | None:
    """将 .env 中的 Langfuse 配置暴露给 Langfuse SDK。"""
    settings = settings or default_settings()
    if settings is None:
        return None
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    if settings.langfuse_base_url:
        os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)
    return settings


def create_trace_id(seed: str, settings: DeclarationQASettings | None = None) -> str:
    """创建确定性的 Langfuse trace ID，供后续反馈评分使用。"""
    if not langfuse_ready(settings):
        return ""
    try:
        from langfuse import Langfuse

        return str(Langfuse.create_trace_id(seed=str(seed or "")))
    except Exception:
        return ""


@contextmanager
def observation(
    name: str,
    *,
    settings: DeclarationQASettings | None = None,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    output: Any = None,
    trace_id: str = "",
    user_id: str = "",
    session_id: str = "",
    tags: list[str] | None = None,
    trace_name: str = "",
    as_type: str = "span",
    model: str = "",
):
    """配置可用时启动 Langfuse 观测，否则返回空实现。"""
    if not langfuse_ready(settings):
        yield NoopObservation()
        return
    stack = ExitStack()
    try:
        from langfuse import get_client, propagate_attributes

        client = get_client()
        metadata_payload = safe_value(metadata or {})
        effective_trace_id = trace_id or trace_id_from_metadata(metadata_payload) or CURRENT_TRACE_ID.get()
        trace_depth = CURRENT_TRACE_DEPTH.get()
        kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
        if model:
            kwargs["model"] = model
        if effective_trace_id and trace_depth <= 0:
            kwargs["trace_context"] = {"trace_id": effective_trace_id}
        raw = stack.enter_context(client.start_as_current_observation(**kwargs))
        active_trace_id = effective_trace_id or str(getattr(raw, "trace_id", "") or "")
        active_span_id = str(getattr(raw, "id", "") or "")
        if active_trace_id:
            token = CURRENT_TRACE_ID.set(active_trace_id)
            stack.callback(reset_current_trace_id, token)
        if active_span_id:
            span_token = CURRENT_SPAN_ID.set(active_span_id)
            stack.callback(reset_current_span_id, span_token)
        depth_token = CURRENT_TRACE_DEPTH.set(trace_depth + 1)
        stack.callback(reset_current_trace_depth, depth_token)
        if trace_name or user_id or session_id or tags:
            stack.enter_context(
                propagate_attributes(
                    trace_name=trace_name or None,
                    user_id=user_id or None,
                    session_id=session_id or None,
                    tags=tags or None,
                    metadata=metadata_payload,
                )
            )
    except Exception:
        stack.close()
        yield NoopObservation()
        return

    wrapped = Observation(raw, metadata=metadata_payload, trace_id=active_trace_id)
    try:
        wrapped.update(input=safe_value(input), output=safe_value(output), metadata=metadata_payload)
        yield wrapped
    except Exception as exc:
        record_error(wrapped, exc, stage=name)
        raise
    finally:
        stack.close()


class Observation:
    """Langfuse 观测对象的安全包装。"""

    def __init__(self, raw: Any, *, metadata: dict[str, Any] | None = None, trace_id: str = "") -> None:
        self.raw = raw
        self.trace_id = str(trace_id or getattr(raw, "trace_id", "") or "")
        self.metadata = metadata or {}

    def update(self, **kwargs: Any) -> None:
        """更新当前观测，并阻止 SDK 异常外泄。"""
        payload = {key: value for key, value in kwargs.items() if value not in (None, "", [], {})}
        if not payload:
            return
        if isinstance(payload.get("metadata"), dict):
            self.metadata = {**self.metadata, **payload["metadata"]}
            payload["metadata"] = self.metadata
        try:
            self.raw.update(**payload)
        except Exception:
            return

    def set_trace_io(self, **kwargs: Any) -> None:
        """在 SDK 支持时设置 trace 输入输出。"""
        payload = {key: value for key, value in kwargs.items() if value not in (None, "", [], {})}
        if not payload:
            return
        try:
            self.raw.set_trace_io(**payload)
        except Exception:
            return

    def score_trace(self, **kwargs: Any) -> None:
        """在 SDK 支持时为当前 trace 评分。"""
        try:
            self.raw.score_trace(**kwargs)
        except Exception:
            return

    def record_error(self, exc: BaseException | None = None, **kwargs: Any) -> None:
        """在观测对象上记录错误载荷。"""
        record_error(self, exc, **kwargs)


def error_payload(
    exc: BaseException | None = None,
    *,
    stage: str = "",
    reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建脱敏后的错误载荷。"""
    payload = {
        "ok": False,
        "stage": stage,
        "reason": reason or (type(exc).__name__ if exc else ""),
        "error_type": type(exc).__name__ if exc else "",
        "message": str(exc or ""),
        **(extra or {}),
    }
    return safe_value(payload)


def record_error(
    span: Any,
    exc: BaseException | None = None,
    *,
    stage: str = "",
    reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """向 Langfuse 观测写入结构化错误载荷。"""
    payload = error_payload(exc, stage=stage, reason=reason, extra=extra)
    metadata = {"error": payload}
    if isinstance(span, Observation):
        metadata = {**span.metadata, "error": payload}
    try:
        span.update(output={"ok": False, "error": payload}, metadata=metadata)
    except Exception:
        return payload
    return payload


def trace_id_from_metadata(metadata: Any) -> str:
    """从观测元数据中读取 Langfuse trace ID。"""
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("langfuse_trace_id") or metadata.get("trace_id") or "").strip()


def reset_current_trace_id(token: Token[str]) -> None:
    """退出观测时恢复当前 trace 上下文。"""
    try:
        CURRENT_TRACE_ID.reset(token)
    except Exception:
        return


def reset_current_trace_depth(token: Token[int]) -> None:
    """退出观测时恢复当前 trace 嵌套深度。"""
    try:
        CURRENT_TRACE_DEPTH.reset(token)
    except Exception:
        return


def reset_current_span_id(token: Token[str]) -> None:
    """退出观测时恢复当前 span 上下文。"""
    try:
        CURRENT_SPAN_ID.reset(token)
    except Exception:
        return


def langchain_config(state: dict[str, Any], *, run_name: str, tags: list[str] | None = None) -> dict[str, Any] | None:
    """构建 Langfuse 使用的 LangChain 回调配置。"""
    if not langfuse_ready():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        return None
    metadata = {
        "langfuse_user_id": str(state.get("user_id") or ""),
        "langfuse_session_id": str(state.get("session_id") or ""),
        "route": state.get("route") or "",
        "intent": state.get("intent") or "",
        "ticket_fact_type": state.get("ticket_fact_type") or "none",
        "retrieval_chain": state.get("retrieval_chain") or "",
    }
    trace_id = str(state.get("langfuse_trace_id") or "")
    if trace_id:
        metadata["langfuse_trace_id"] = trace_id
    trace_context = {}
    if trace_id and CURRENT_TRACE_DEPTH.get() <= 0:
        trace_context["trace_id"] = trace_id
    handler = CallbackHandler(trace_context=trace_context) if trace_context else CallbackHandler()
    return {
        "callbacks": [handler],
        "run_name": run_name,
        "tags": tags or ["declaration-qa"],
        "metadata": safe_value(metadata),
    }


def invoke_with_langfuse(llm: Any, prompt: Any, state: dict[str, Any], *, run_name: str) -> Any:
    """启用时通过 Langfuse 回调调用 LangChain LLM。"""
    config = langchain_config(state, run_name=run_name)
    if config:
        return llm.invoke(prompt, config=config)
    return llm.invoke(prompt)


def stream_with_langfuse(llm: Any, prompt: Any, state: dict[str, Any], *, run_name: str) -> Any:
    """启用时通过 Langfuse 回调流式调用 LangChain LLM。"""
    config = langchain_config(state, run_name=run_name)
    if config:
        return llm.stream(prompt, config=config)
    return llm.stream(prompt)


def score_trace(
    trace_id: str,
    *,
    name: str,
    value: float | int,
    comment: str = "",
    settings: DeclarationQASettings | None = None,
) -> None:
    """根据 trace ID 创建 Langfuse trace 评分。"""
    if not trace_id or not langfuse_ready(settings):
        return
    try:
        from langfuse import get_client

        get_client().create_score(
            trace_id=str(trace_id),
            name=name,
            value=value,
            data_type="NUMERIC",
            comment=comment or None,
        )
    except Exception:
        return


def source_summary(sources: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    """返回用于可观测性的紧凑检索来源摘要。"""
    output = []
    for source in (sources or [])[:limit]:
        output.append(
            {
                "chunk_id": source.get("chunk_id") or "",
                "doc_id": source.get("doc_id") or "",
                "title": source.get("title") or "",
                "score": source.get("score") or 0,
                "rerank_score": source.get("rerank_score") or 0,
                "channel": source.get("retrieval_channel") or "",
                "item_name": (source.get("metadata") or {}).get("item_name") or source.get("item_name") or "",
            }
        )
    return output


def retrieved_source_summary(sources: list[Any], *, limit: int = 8) -> list[dict[str, Any]]:
    """返回 RetrievedSource 类对象的紧凑摘要。"""
    output = []
    for source in (sources or [])[:limit]:
        chunk = getattr(source, "chunk", None)
        metadata = getattr(chunk, "metadata", {}) or {}
        output.append(
            {
                "chunk_id": getattr(chunk, "id", "") or "",
                "doc_id": getattr(chunk, "doc_id", "") or "",
                "title": getattr(chunk, "title", "") or "",
                "score": round(float(getattr(source, "score", 0.0) or 0.0), 4),
                "dense_score": round(float(getattr(source, "dense_score", 0.0) or 0.0), 4),
                "bm25_score": round(float(getattr(source, "bm25_score", 0.0) or 0.0), 4),
                "rerank_score": round(float(getattr(source, "rerank_score", 0.0) or 0.0), 4),
                "channel": getattr(source, "retrieval_channel", "") or "",
                "item_name": metadata.get("item_name") or "",
                "chunk_type": getattr(chunk, "chunk_type", "") or "",
            }
        )
    return output


def safe_value(value: Any, *, depth: int = 0) -> Any:
    """发送到可观测性后端前对值进行脱敏和截断。"""
    if depth > 4:
        return str(type(value).__name__)
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_DICT_ITEMS:
                output["..."] = f"{len(value) - MAX_DICT_ITEMS} more"
                break
            key_text = str(key)
            if is_sensitive_key(key_text):
                output[key_text] = "[redacted]"
            else:
                output[key_text] = safe_value(item, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        output = [safe_value(item, depth=depth + 1) for item in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            output.append(f"... {len(items) - MAX_LIST_ITEMS} more")
        return output
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING_LENGTH else f"{value[:MAX_STRING_LENGTH]}..."
    return value


def is_sensitive_key(key: str) -> bool:
    """返回元数据键是否需要脱敏。"""
    lowered = key.lower()
    return any(token in lowered for token in SENSITIVE_KEYWORDS) or lowered in {"name", "applicant_name"}
