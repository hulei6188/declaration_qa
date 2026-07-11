"""审计和反馈路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from declaration_qa.api.auth import RequestContext, require_service_or_user
from declaration_qa.api.schemas import AgentRunDetail, FeedbackRequest
from declaration_qa.api.server.dependencies import auth_context
from declaration_qa.api.server.runtime import settings, workflow
from declaration_qa.observability.langfuse import score_trace


def register_audit_routes(app):
    """注册审计和反馈路由。"""
    @app.get("/api/declaration-qa/agent-runs/{run_id}", response_model=AgentRunDetail)
    def get_agent_run(
        run_id: str,
        user_id: int = 1,
        context: RequestContext = Depends(auth_context),
    ):
        """按用户读取完整且有序的 Agent 消息和工具调用。"""
        require_service_or_user(context, settings)
        effective_user_id = context.user_id or user_id
        getter = getattr(workflow.audit_store, "get_agent_run", None)
        if not callable(getter):
            raise HTTPException(status_code=501, detail="Agent run replay is not supported")
        item = getter(user_id=int(effective_user_id), run_id=run_id)
        if not item:
            raise HTTPException(status_code=404, detail="Agent run not found")
        return item

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
