"""审计和反馈路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from gov_service_agent.api.auth import RequestContext, require_service_or_user
from gov_service_agent.api.schemas import FeedbackRequest
from gov_service_agent.api.server.dependencies import auth_context
from gov_service_agent.api.server.runtime import settings, workflow
from gov_service_agent.observability.langfuse import score_trace


def register_audit_routes(app):
    """注册审计和反馈路由。"""
    @app.get("/api/declaration-qa/audit/{audit_id}")
    def get_audit(audit_id: str, context: RequestContext = Depends(auth_context)):
        """查询并返回对应的数据详情。"""
        require_service_or_user(context, settings)
        item = workflow.audit_store.get(audit_id)
        if not item:
            raise HTTPException(status_code=404, detail="Audit log not found")
        return item

    @app.post("/api/declaration-qa/feedback")
    def feedback(request: FeedbackRequest, context: RequestContext = Depends(auth_context)):
        """执行当前函数对应的业务处理。"""
        require_service_or_user(context, settings)
        audit = workflow.audit_store.get(request.audit_id)
        if not audit:
            raise HTTPException(status_code=404, detail="Audit log not found")
        result = workflow.audit_store.add_feedback(
            audit_id=request.audit_id,
            rating=request.rating,
            comment=request.comment,
        )
        score_trace(
            str(audit.get("langfuse_trace_id") or ""),
            name="user_feedback",
            value=1 if request.rating == "up" else 0,
            comment=request.comment or "",
            settings=settings,
        )
        return result
