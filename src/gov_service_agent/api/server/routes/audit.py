"""审计和反馈路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from gov_service_agent.api.auth import RequestContext, require_service_or_user
from gov_service_agent.api.schemas import AgentRunDetail, FeedbackRequest, HumanServiceRequest, HumanServiceTicketResponse
from gov_service_agent.api.server.dependencies import auth_context
from gov_service_agent.api.server.runtime import settings, workflow
from gov_service_agent.observability.langfuse import score_trace


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

    @app.post("/api/declaration-qa/human-service/tickets", response_model=HumanServiceTicketResponse)
    def create_human_service_ticket(request: HumanServiceRequest, context: RequestContext = Depends(auth_context)):
        require_service_or_user(context, settings)
        effective_user_id = int(context.user_id or request.user_id)
        audit = workflow.audit_store.get(request.audit_id)
        if not audit:
            raise HTTPException(status_code=404, detail="Audit record not found")
        if int(audit.get("user_id") or effective_user_id) != effective_user_id:
            raise HTTPException(status_code=403, detail="Ticket does not belong to current user")
        try:
            return workflow.audit_store.create_human_service_ticket(
                audit_id=request.audit_id,
                user_id=effective_user_id,
                session_id=request.session_id or str(audit.get("session_id") or ""),
                note=request.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
