"""服务依赖函数。"""

from __future__ import annotations

from fastapi import Request

from declaration_qa.api.auth import RequestContext, resolve_request_context
from declaration_qa.api.server.runtime import settings


def auth_context(request: Request) -> RequestContext:
    """解析当前请求的认证上下文。"""
    return resolve_request_context(request, settings)
