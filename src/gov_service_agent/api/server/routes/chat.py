"""Agentic RAG 聊天路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from gov_service_agent.api.auth import RequestContext, apply_chat_auth
from gov_service_agent.api.schemas import ChatRequest, ChatResponse, HumanConfirmationRequest
from gov_service_agent.api.server.chat_images import prepare_chat_image_request
from gov_service_agent.api.server.dependencies import auth_context
from gov_service_agent.api.server.knowledge_admin import ensure_knowledge_base_fresh
from gov_service_agent.api.server.runtime import settings, workflow
from gov_service_agent.api.server.sse import SSE_HEADERS, threaded_sse_event_stream


def register_chat_routes(app):
    """注册非流式和流式聊天路由。"""
    @app.post("/api/declaration-qa/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, context: RequestContext = Depends(auth_context)):
        """处理非流式问答请求。"""
        ensure_knowledge_base_fresh()
        prepared = prepare_chat_image_request(request, settings)
        return workflow.invoke(apply_chat_auth(prepared, context, settings))

    @app.post("/api/declaration-qa/chat/stream")
    def chat_stream(request: ChatRequest, context: RequestContext = Depends(auth_context)):
        """处理流式问答请求。"""
        ensure_knowledge_base_fresh()
        prepared = prepare_chat_image_request(request, settings)
        return StreamingResponse(
            threaded_sse_event_stream(workflow.stream(apply_chat_auth(prepared, context, settings))),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.post("/api/declaration-qa/chat/confirm/stream")
    def confirm_chat_action(
        request: HumanConfirmationRequest,
        context: RequestContext = Depends(auth_context),
    ):
        """确认或拒绝暂停中的预约写操作，并继续原 Agent 运行。"""
        user_id = int(context.user_id or request.user_id)
        if settings.effective_auth_required and not context.user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated user_id is required",
            )
        prepared = request.model_copy(update={"user_id": user_id})

        def events():
            try:
                yield from workflow.resume_stream(prepared)
            except ValueError as exc:
                yield {"event": "error", "data": {"message": str(exc)}}

        return StreamingResponse(
            threaded_sse_event_stream(events()),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
