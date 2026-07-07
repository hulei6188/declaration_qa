"""Langfuse observability helpers."""

from __future__ import annotations

import os
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


class NoopObservation:
    """No-op object with the Langfuse observation methods used by this app."""

    trace_id = ""

    def update(self, **_: Any) -> None:
        """Ignore updates when observability is disabled."""

    def set_trace_io(self, **_: Any) -> None:
        """Ignore trace I/O updates when observability is disabled."""

    def score_trace(self, **_: Any) -> None:
        """Ignore trace scores when observability is disabled."""

    def record_error(self, *_: Any, **__: Any) -> None:
        """Ignore errors when observability is disabled."""


def langfuse_ready(settings: DeclarationQASettings | None = None) -> bool:
    """Return whether Langfuse tracing should be attempted."""
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
    """Load project settings from environment and .env for observability helpers."""
    try:
        return DeclarationQASettings()
    except Exception:
        return None


def ensure_langfuse_env(settings: DeclarationQASettings | None = None) -> DeclarationQASettings | None:
    """Expose .env Langfuse settings to the Langfuse SDK."""
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
    """Create a deterministic Langfuse trace id for later feedback scoring."""
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
    as_type: str = "span",
    model: str = "",
):
    """Start a Langfuse observation if configured, otherwise yield a no-op."""
    if not langfuse_ready(settings):
        yield NoopObservation()
        return
    stack = ExitStack()
    try:
        from langfuse import get_client, propagate_attributes

        client = get_client()
        kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
        if model:
            kwargs["model"] = model
        if trace_id:
            kwargs["trace_context"] = {"trace_id": trace_id}
        raw = stack.enter_context(client.start_as_current_observation(**kwargs))
        stack.enter_context(
            propagate_attributes(
                trace_name=name,
                user_id=user_id or None,
                session_id=session_id or None,
                tags=tags or None,
                metadata=safe_value(metadata or {}),
            )
        )
    except Exception:
        stack.close()
        yield NoopObservation()
        return

    wrapped = Observation(raw, metadata=safe_value(metadata or {}))
    try:
        wrapped.update(input=safe_value(input), output=safe_value(output), metadata=safe_value(metadata or {}))
        yield wrapped
    except Exception as exc:
        record_error(wrapped, exc, stage=name)
        raise
    finally:
        stack.close()


class Observation:
    """Safe wrapper around a Langfuse observation."""

    def __init__(self, raw: Any, *, metadata: dict[str, Any] | None = None) -> None:
        self.raw = raw
        self.trace_id = str(getattr(raw, "trace_id", "") or getattr(raw, "id", "") or "")
        self.metadata = metadata or {}

    def update(self, **kwargs: Any) -> None:
        """Update the current observation without letting SDK errors escape."""
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
        """Set trace input/output when supported by the SDK."""
        payload = {key: value for key, value in kwargs.items() if value not in (None, "", [], {})}
        if not payload:
            return
        try:
            self.raw.set_trace_io(**payload)
        except Exception:
            return

    def score_trace(self, **kwargs: Any) -> None:
        """Score current trace when supported by the SDK."""
        try:
            self.raw.score_trace(**kwargs)
        except Exception:
            return

    def record_error(self, exc: BaseException | None = None, **kwargs: Any) -> None:
        """Record an error payload on the observation."""
        record_error(self, exc, **kwargs)


def error_payload(
    exc: BaseException | None = None,
    *,
    stage: str = "",
    reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sanitized error payload."""
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
    """Write a structured error payload to a Langfuse observation."""
    payload = error_payload(exc, stage=stage, reason=reason, extra=extra)
    metadata = {"error": payload}
    if isinstance(span, Observation):
        metadata = {**span.metadata, "error": payload}
    try:
        span.update(output={"ok": False, "error": payload}, metadata=metadata)
    except Exception:
        return payload
    return payload


def langchain_config(state: dict[str, Any], *, run_name: str, tags: list[str] | None = None) -> dict[str, Any] | None:
    """Build LangChain callback config for Langfuse."""
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
    return {
        "callbacks": [CallbackHandler()],
        "run_name": run_name,
        "tags": tags or ["declaration-qa"],
        "metadata": safe_value(metadata),
    }


def invoke_with_langfuse(llm: Any, prompt: str, state: dict[str, Any], *, run_name: str) -> Any:
    """Invoke a LangChain LLM with Langfuse callbacks when enabled."""
    config = langchain_config(state, run_name=run_name)
    if config:
        return llm.invoke(prompt, config=config)
    return llm.invoke(prompt)


def stream_with_langfuse(llm: Any, prompt: str, state: dict[str, Any], *, run_name: str) -> Any:
    """Stream a LangChain LLM with Langfuse callbacks when enabled."""
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
    """Create a Langfuse trace score by trace id."""
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
    """Return a compact retrieval source summary for observability."""
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
    """Return a compact summary for RetrievedSource-like objects."""
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
    """Redact and truncate values before sending them to observability sinks."""
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
    """Return whether a metadata key should be redacted."""
    lowered = key.lower()
    return any(token in lowered for token in SENSITIVE_KEYWORDS) or lowered in {"name", "applicant_name"}
