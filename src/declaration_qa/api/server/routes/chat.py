"""聊天和办件查询路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

from declaration_qa.api.auth import RequestContext, apply_chat_auth
from declaration_qa.api.schemas import ChatRequest, ChatResponse, TicketSummary
from declaration_qa.api.server.dependencies import auth_context
from declaration_qa.api.server.knowledge_admin import ensure_knowledge_base_fresh
from declaration_qa.api.server.runtime import settings, workflow
from declaration_qa.api.server.sse import SSE_HEADERS, threaded_sse_event_stream


def register_chat_routes(app):
    """注册聊天和办件查询路由。"""
    @app.post("/api/declaration-qa/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, context: RequestContext = Depends(auth_context)):
        """处理非流式问答请求。"""
        ensure_knowledge_base_fresh()
        return workflow.invoke(apply_chat_auth(request, context, settings))

    @app.post("/api/declaration-qa/chat/stream")
    def chat_stream(request: ChatRequest, context: RequestContext = Depends(auth_context)):
        """处理流式问答请求。"""
        ensure_knowledge_base_fresh()
        return StreamingResponse(
            threaded_sse_event_stream(workflow.stream(apply_chat_auth(request, context, settings))),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.get("/api/declaration-qa/tickets", response_model=list[TicketSummary])
    def list_tickets(
        user_id: int = 1,
        biz_type: str = "",
        region_code: str = "340000",
        context: RequestContext = Depends(auth_context),
    ):
        """查询当前用户办件列表。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        if not hasattr(workflow.business_adapter, "list_applications"):
            return []
        return workflow.business_adapter.list_applications(
            user_id=int(effective_user_id or 1),
            biz_type=biz_type or None,
            region_code=region_code or None,
        )
