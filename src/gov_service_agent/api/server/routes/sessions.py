"""历史会话路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from gov_service_agent.api.auth import RequestContext
from gov_service_agent.api.schemas import DeleteSessionResponse, SessionDetail, SessionSummary
from gov_service_agent.api.server.dependencies import auth_context
from gov_service_agent.api.server.runtime import settings, workflow


def register_session_routes(app):
    """注册历史会话路由。"""
    @app.get("/api/declaration-qa/sessions", response_model=list[SessionSummary])
    def list_sessions(user_id: int = 1, context: RequestContext = Depends(auth_context)):
        """按用户查询历史会话摘要列表。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authenticated user_id is required")
        return workflow.audit_store.list_sessions(user_id=int(effective_user_id))

    @app.get("/api/declaration-qa/sessions/{session_id}", response_model=SessionDetail)
    def get_session(session_id: str, user_id: int = 1, context: RequestContext = Depends(auth_context)):
        """按用户和会话 ID 查询完整会话详情。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authenticated user_id is required")
        item = workflow.audit_store.get_session(user_id=int(effective_user_id), session_id=session_id)
        if not item:
            raise HTTPException(status_code=404, detail="Session not found")
        return item

    @app.delete("/api/declaration-qa/sessions/{session_id}", response_model=DeleteSessionResponse)
    def delete_session(session_id: str, user_id: int = 1, context: RequestContext = Depends(auth_context)):
        """按用户删除指定历史会话。"""
        effective_user_id = context.user_id or user_id
        if settings.effective_auth_required and not effective_user_id:
            raise HTTPException(status_code=401, detail="Authenticated user_id is required")
        deleted = workflow.delete_session(user_id=int(effective_user_id), session_id=session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return DeleteSessionResponse(session_id=session_id, deleted=True)
