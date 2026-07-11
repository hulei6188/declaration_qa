"""Agentic RAG 聊天路由。"""

from __future__ import annotations

from fastapi import Depends
from fastapi.responses import StreamingResponse

from declaration_qa.api.auth import RequestContext, apply_chat_auth
from declaration_qa.api.schemas import ChatRequest, ChatResponse
from declaration_qa.api.server.chat_images import prepare_chat_image_request
from declaration_qa.api.server.dependencies import auth_context
from declaration_qa.api.server.knowledge_admin import ensure_knowledge_base_fresh
from declaration_qa.api.server.runtime import settings, workflow
from declaration_qa.api.server.sse import SSE_HEADERS, threaded_sse_event_stream


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
