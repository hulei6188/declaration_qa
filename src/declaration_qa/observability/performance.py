"""Lightweight structured performance logging for request and retrieval stages."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Any, Iterator


# Use Uvicorn's configured handler so INFO records are visible in the API
# terminal without requiring a separate application-wide logging config.
LOGGER = logging.getLogger("uvicorn.error.declaration_qa.performance")
LOGGER.setLevel(logging.INFO)

_PERFORMANCE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "declaration_qa_performance_context",
    default={},
)


def set_performance_context(**fields: Any) -> Token:
    """Bind correlation fields to the current request context."""
    current = dict(_PERFORMANCE_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value not in (None, "")})
    return _PERFORMANCE_CONTEXT.set(current)


def reset_performance_context(token: Token) -> None:
    """Restore the performance context that preceded ``set_performance_context``."""
    _PERFORMANCE_CONTEXT.reset(token)


@contextmanager
def performance_request(stage: str, **fields: Any) -> Iterator[None]:
    """Bind request correlation fields and log the full request duration."""
    token = set_performance_context(**fields)
    try:
        with performance_stage(stage):
            yield
    finally:
        reset_performance_context(token)


@contextmanager
def performance_stage(stage: str, **fields: Any) -> Iterator[None]:
    """Log one stage with elapsed time and exception status."""
    started_at = perf_counter()
    try:
        yield
    except Exception as exc:
        log_performance(
            stage,
            started_at,
            status="error",
            error_type=type(exc).__name__,
            **fields,
        )
        raise
    else:
        log_performance(stage, started_at, **fields)


def log_performance(
    stage: str,
    started_at: float,
    *,
    status: str = "ok",
    **fields: Any,
) -> None:
    """Emit a single JSON log record without request bodies or credentials."""
    payload = {
        "event": "performance",
        "stage": str(stage),
        "status": str(status),
        "elapsed_ms": max(0, round((perf_counter() - started_at) * 1000)),
        **_PERFORMANCE_CONTEXT.get(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    LOGGER.info("%s", json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")))
