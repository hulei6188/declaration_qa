"""坐席人工服务工作台路由。"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query

from gov_service_agent.api.auth import RequestContext, require_evaluation_admin
from gov_service_agent.api.schemas import (
    HumanServiceMessageRequest,
    HumanServiceStatusRequest,
    HumanServiceTicketResponse,
)
from gov_service_agent.api.server.dependencies import auth_context
from gov_service_agent.api.server.runtime import settings, workflow


def register_human_service_routes(app):
    @app.get("/api/declaration-qa/human-service/tickets", response_model=list[HumanServiceTicketResponse])
    def list_tickets(status: str = Query(default="", pattern="^(|open|claimed|waiting_user|resolved|closed)$"), limit: int = Query(default=100, ge=1, le=500), context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        return workflow.audit_store.list_human_service_tickets(status=status, limit=limit)

    @app.get("/api/declaration-qa/human-service/tickets/{ticket_id}", response_model=HumanServiceTicketResponse)
    def get_ticket(ticket_id: str, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        ticket = workflow.audit_store.get_human_service_ticket(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Human service ticket not found")
        return ticket

    @app.post("/api/declaration-qa/human-service/tickets/{ticket_id}/claim", response_model=HumanServiceTicketResponse)
    def claim_ticket(ticket_id: str, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.claim_human_service_ticket(ticket_id=ticket_id, agent_id=int(context.user_id or 0))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/declaration-qa/human-service/tickets/{ticket_id}/status", response_model=HumanServiceTicketResponse)
    def update_ticket_status(ticket_id: str, request: HumanServiceStatusRequest, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.update_human_service_status(ticket_id=ticket_id, status=request.status)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/declaration-qa/human-service/tickets/{ticket_id}/messages", response_model=HumanServiceTicketResponse)
    def add_ticket_message(ticket_id: str, request: HumanServiceMessageRequest, context: RequestContext = Depends(auth_context)):
        require_evaluation_admin(context, settings)
        try:
            return workflow.audit_store.add_human_service_message(ticket_id=ticket_id, role="agent", content=request.content, author_id=int(context.user_id or 0))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
