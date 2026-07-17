"""用于请求与检索阶段的轻量结构化性能日志。"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Any, Iterator


# 复用 Uvicorn 已配置的处理器，让信息级日志直接显示在 API 终端，
# 无需额外配置全局日志系统。
LOGGER = logging.getLogger("uvicorn.error.gov_service_agent.performance")
LOGGER.setLevel(logging.INFO)

_PERFORMANCE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "declaration_qa_performance_context",
    default={},
)


def set_performance_context(**fields: Any) -> Token:
    """将关联字段绑定到当前请求上下文。"""
    current = dict(_PERFORMANCE_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value not in (None, "")})
    return _PERFORMANCE_CONTEXT.set(current)


def reset_performance_context(token: Token) -> None:
    """恢复调用 ``set_performance_context`` 之前的性能日志上下文。"""
    _PERFORMANCE_CONTEXT.reset(token)


@contextmanager
def performance_request(stage: str, **fields: Any) -> Iterator[None]:
    """绑定请求关联字段并记录完整请求耗时。"""
    token = set_performance_context(**fields)
    try:
        with performance_stage(stage):
            yield
    finally:
        reset_performance_context(token)


@contextmanager
def performance_stage(stage: str, **fields: Any) -> Iterator[None]:
    """记录单个阶段的耗时和异常状态。"""
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
    """输出一条不包含请求正文或凭据的 JSON 日志。"""
    payload = {
        "event": "performance",
        "stage": str(stage),
        "status": str(status),
        "elapsed_ms": max(0, round((perf_counter() - started_at) * 1000)),
        **_PERFORMANCE_CONTEXT.get(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    LOGGER.info("%s", json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")))
