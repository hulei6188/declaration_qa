"""服务依赖函数。"""

from __future__ import annotations

from fastapi import Request

from gov_service_agent.api.auth import RequestContext, resolve_request_context
from gov_service_agent.api.server.runtime import settings


def auth_context(request: Request) -> RequestContext:
    """解析当前请求的认证上下文。"""
    return resolve_request_context(request, settings)
