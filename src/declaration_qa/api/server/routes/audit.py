"""审计和反馈路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from declaration_qa.api.auth import RequestContext, require_service_or_user
from declaration_qa.api.schemas import FeedbackRequest
from declaration_qa.api.server.dependencies import auth_context
from declaration_qa.api.server.runtime import settings, workflow


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
        if not workflow.audit_store.get(request.audit_id):
            raise HTTPException(status_code=404, detail="Audit log not found")
        return workflow.audit_store.add_feedback(
            audit_id=request.audit_id,
            rating=request.rating,
            comment=request.comment,
        )
