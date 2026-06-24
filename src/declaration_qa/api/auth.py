"""请求认证、JWT 解析和用户上下文处理。"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt

from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.api.schemas import ChatRequest


@dataclass(frozen=True)
class RequestContext:
    """当前请求解析后的认证与用户上下文。"""
    user_id: int | None = None
    subject: str = ""
    scopes: tuple[str, ...] = ()
    auth_type: str = "anonymous"
    service: bool = False


def resolve_request_context(request: Request, settings: DeclarationQASettings) -> RequestContext:
    """从请求头解析认证方式和用户上下文。"""
    authorization = request.headers.get("Authorization", "")
    api_key = request.headers.get("X-Declaration-QA-API-Key", "")

    if api_key:
        if not settings.internal_api_key or api_key != settings.internal_api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal API key")
        user_id = parse_int_header(request.headers.get("X-User-Id"))
        return RequestContext(user_id=user_id, subject="internal-api-key", auth_type="api_key", service=True)

    if authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        return decode_jwt_context(token, settings)

    if settings.effective_auth_required:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization is required")

    return RequestContext()


def decode_jwt_context(token: str, settings: DeclarationQASettings) -> RequestContext:
    """解码 JWT 并转换为请求上下文。"""
    if not settings.jwt_secret:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JWT secret is not configured")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=settings.jwt_algorithm_list)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token") from exc

    subject = str(payload.get("sub") or payload.get("user_id") or payload.get("uid") or "")
    user_id = parse_int(payload.get("user_id") or payload.get("uid") or payload.get("sub"))
    scopes = payload.get("scope") or payload.get("scopes") or []
    if isinstance(scopes, str):
        scope_tuple = tuple(item for item in scopes.split(" ") if item)
    else:
        scope_tuple = tuple(str(item) for item in scopes)
    return RequestContext(user_id=user_id, subject=subject, scopes=scope_tuple, auth_type="jwt")


def apply_chat_auth(request: ChatRequest, context: RequestContext, settings: DeclarationQASettings) -> ChatRequest:
    """将认证用户写入问答请求或执行鉴权校验。"""
    if not context.user_id:
        if settings.effective_auth_required:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user_id is required")
        return request
    return request.model_copy(update={"user_id": context.user_id})


def require_service_or_user(context: RequestContext, settings: DeclarationQASettings) -> None:
    """校验当前请求具备服务或用户身份。"""
    if settings.effective_auth_required and not (context.service or context.user_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user or service is required")


def parse_int_header(value: str | None) -> int | None:
    """将请求头字符串安全转换为整数。"""
    return parse_int(value)


def parse_int(value) -> int | None:
    """将任意输入安全转换为整数。"""
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
