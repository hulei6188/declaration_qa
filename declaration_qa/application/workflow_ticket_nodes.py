"""Ticket and business-context nodes for the declaration QA workflow."""

from __future__ import annotations

from declaration_qa.application.state import DeclarationQAState
from declaration_qa.application.workflow_helpers import (
    add_event,
    base_policy_filter,
    build_business_snapshot,
    material_tags,
    ticket_summary_from_context,
)
from declaration_qa.rag.router import INTENT_REJECTION_REASON, INTENT_TICKET_STATUS


class WorkflowTicketMixin:
    """Ticket selection and business-data enrichment nodes."""
    def ticket_list_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询用户办件列表并写入状态。"""
        next_state = dict(state)
        if next_state.get("apply_context"):
            next_state["tickets"] = [ticket_summary_from_context(next_state["apply_context"] or {})]
            add_event(next_state, "ticket_list", {"count": 1, "source": "apply_context"})
            return next_state
        if not hasattr(self.business_adapter, "list_applications"):
            next_state["tickets"] = []
            next_state["refuse_reason"] = "ticket_not_found"
            add_event(next_state, "ticket_list", {"count": 0, "source": "adapter_missing"})
            return next_state
        tickets = self.business_adapter.list_applications(
            user_id=int(next_state["user_id"]),
            biz_type=next_state.get("biz_type"),
            region_code=next_state.get("region_code"),
        )
        if not tickets and next_state.get("biz_type"):
            tickets = self.business_adapter.list_applications(
                user_id=int(next_state["user_id"]),
                biz_type=None,
                region_code=next_state.get("region_code"),
            )
        next_state["tickets"] = tickets
        if not tickets:
            next_state["refuse_reason"] = "ticket_not_found"
        add_event(next_state, "ticket_list", {"count": len(tickets), "biz_type": next_state.get("biz_type")})
        return next_state

    def ticket_selection(self, state: DeclarationQAState) -> DeclarationQAState:
        """选择当前问题对应的办件。"""
        next_state = dict(state)
        if next_state.get("apply_context"):
            next_state["need_ticket_selection"] = False
            next_state["ticket_selection_reason"] = "provided_apply_context"
            add_event(next_state, "ticket_selected", {"reason": "provided_apply_context"})
            return next_state
        tickets = next_state.get("tickets") or []
        if len(tickets) == 1:
            ticket_id = str(tickets[0].get("ticket_id") or "")
            apply_context = self.business_adapter.get_apply_context(user_id=int(next_state["user_id"]), apply_id=ticket_id)
            if apply_context:
                next_state["apply_id"] = ticket_id
                next_state["apply_context"] = apply_context
                next_state["need_ticket_selection"] = False
                next_state["ticket_selection_reason"] = "single_candidate"
                add_event(next_state, "ticket_selected", {"ticket_id": ticket_id, "reason": "single_candidate"})
                return next_state
        if len(tickets) > 1:
            next_state["need_ticket_selection"] = True
            next_state["ticket_selection_reason"] = "multiple_candidates"
            next_state["refuse_reason"] = "ticket_selection_required"
            add_event(next_state, "ticket_selection_required", {"count": len(tickets)})
            return next_state
        next_state["need_ticket_selection"] = False
        next_state["ticket_selection_reason"] = "no_candidate"
        next_state["refuse_reason"] = "ticket_not_found"
        add_event(next_state, "ticket_selection_missing", {"reason": "no_candidate"})
        return next_state

    def ticket_detail_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询选中办件的详细上下文。"""
        next_state = dict(state)
        apply_context = next_state.get("apply_context") or {}
        if not apply_context:
            next_state["refuse_reason"] = "ticket_not_found"
            return next_state
        if next_state.get("intent") == INTENT_TICKET_STATUS:
            status = self.business_adapter.get_status(apply_context)
            if not (status.get("status") or status.get("current_node") or status.get("node_name")):
                next_state["business_status"] = {}
                next_state["business_snapshot"] = {}
                next_state["refuse_reason"] = "business_status_not_found"
                add_event(next_state, "business_status_missing", {"apply_id": apply_context.get("apply_id")})
                return next_state
            next_state["business_status"] = status
            next_state["business_snapshot"] = build_business_snapshot("status", status)
            add_event(next_state, "business_status", status)
            return next_state
        if next_state.get("intent") != INTENT_REJECTION_REASON:
            next_state["business_snapshot"] = build_business_snapshot("ticket_context", apply_context)
            add_event(next_state, "ticket_context_business", next_state["business_snapshot"])
            return next_state
        reject_info = self.business_adapter.get_reject_info(apply_context)
        if not (reject_info.get("reject_code") or reject_info.get("reject_reason") or reject_info.get("material_code")):
            next_state["reject_info"] = {}
            next_state["business_snapshot"] = {}
            next_state["refuse_reason"] = "business_reject_not_found"
            add_event(next_state, "business_reject_missing", {"apply_id": apply_context.get("apply_id")})
            return next_state
        next_state["reject_info"] = reject_info
        next_state["business_snapshot"] = build_business_snapshot("reject", reject_info)
        next_state["material_mapping"] = {
            "material_code": reject_info.get("material_code") or "",
            "material_name": reject_info.get("material_name") or "",
            "reject_code": reject_info.get("reject_code") or "",
            "audit_node": reject_info.get("audit_node") or "",
            "knowledge_tags": material_tags(reject_info),
        }
        add_event(next_state, "business_reject", reject_info)
        return next_state

    def ticket_context_build(self, state: DeclarationQAState) -> DeclarationQAState:
        """构造办件增强检索上下文。"""
        next_state = dict(state)
        ticket_context = {
            "ticket": ticket_summary_from_context(next_state.get("apply_context") or {}),
            "business_snapshot": next_state.get("business_snapshot") or {},
            "material_mapping": next_state.get("material_mapping") or {},
        }
        next_state["ticket_context"] = ticket_context
        add_event(next_state, "ticket_context", ticket_context)
        return next_state

    def ticket_knowledge_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        """围绕办件问题执行知识检索。"""
        next_state = dict(state)
        filters = base_policy_filter(next_state, allow_request_filter=False)
        result = self.multi_retrieve(next_state, filters)
        next_state["retrieval_filter"] = filters
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        if next_state.get("intent") == INTENT_TICKET_STATUS:
            next_state["evidence_ok"] = True
            next_state["evidence_decision"] = {
                "ok": True,
                "reason": "business_status_verified",
                "business_snapshot_present": bool(next_state.get("business_snapshot")),
                "source_count": len(result.sources),
            }
        add_event(next_state, "rag_status", {**result.status, "rag_query": next_state.get("rag_query") or ""})
        return next_state

    def status_apply_context_check(self, state: DeclarationQAState) -> DeclarationQAState:
        """检查状态查询是否具备办件上下文。"""
        next_state = dict(state)
        apply_context = next_state.get("apply_context") or {}
        if not apply_context or not str(apply_context.get("apply_id") or "").strip():
            next_state["refuse_reason"] = "apply_context_required_for_status"
        return next_state

    def reject_apply_context_check(self, state: DeclarationQAState) -> DeclarationQAState:
        """检查退回查询是否具备办件上下文。"""
        next_state = dict(state)
        apply_context = next_state.get("apply_context") or {}
        if not apply_context or not str(apply_context.get("apply_id") or "").strip():
            next_state["refuse_reason"] = "apply_context_required_for_reject"
        return next_state

    def business_status_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询办件状态业务数据。"""
        next_state = dict(state)
        status = self.business_adapter.get_status(next_state.get("apply_context") or {})
        if not (status.get("status") or status.get("current_node") or status.get("node_name")):
            next_state["business_status"] = {}
            next_state["business_snapshot"] = {}
            next_state["refuse_reason"] = "business_status_not_found"
            add_event(next_state, "business_status_missing", {"apply_id": (next_state.get("apply_context") or {}).get("apply_id")})
            return next_state
        next_state["business_status"] = status
        next_state["business_snapshot"] = build_business_snapshot("status", status)
        next_state["biz_type"] = next_state.get("biz_type") or status.get("biz_type")
        next_state["region_code"] = status.get("region_code") or next_state.get("region_code")
        add_event(next_state, "business_status", status)
        return next_state

    def status_explain_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        """检索办件状态解释知识。"""
        next_state = dict(state)
        status = next_state.get("business_status") or {}
        if not status:
            next_state["refused"] = True
            next_state["refuse_reason"] = "business_status_not_found"
            next_state["answer"] = "暂未从办件系统查询到该申请单的实时进度，请确认申请单是否有效，或联系人工客服核验。"
            add_event(next_state, "refused", {"reason": next_state["refuse_reason"]})
            return next_state
        filters = base_policy_filter(next_state, allow_request_filter=False)
        next_state["retrieval_filter"] = filters
        query = f"{status.get('node_name') or status.get('current_node') or ''} {next_state.get('normalized_query') or ''}"
        result = self.rag_engine.retrieve(query, filters, top_k=3)
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        next_state["evidence_ok"] = True
        next_state["evidence_decision"] = {
            "ok": True,
            "reason": "business_status_verified",
            "business_snapshot_present": bool(status),
            "source_count": len(result.sources),
        }
        add_event(next_state, "rag_status", result.status)
        return next_state

    def business_reject_query(self, state: DeclarationQAState) -> DeclarationQAState:
        """查询办件退回补正业务数据。"""
        next_state = dict(state)
        reject_info = self.business_adapter.get_reject_info(next_state.get("apply_context") or {})
        if not (reject_info.get("reject_code") or reject_info.get("reject_reason") or reject_info.get("material_code")):
            next_state["reject_info"] = {}
            next_state["business_snapshot"] = {}
            next_state["refuse_reason"] = "business_reject_not_found"
            add_event(next_state, "business_reject_missing", {"apply_id": (next_state.get("apply_context") or {}).get("apply_id")})
            return next_state
        next_state["reject_info"] = reject_info
        next_state["business_snapshot"] = build_business_snapshot("reject", reject_info)
        next_state["biz_type"] = next_state.get("biz_type") or reject_info.get("biz_type")
        next_state["region_code"] = reject_info.get("region_code") or next_state.get("region_code")
        add_event(next_state, "business_reject", reject_info)
        return next_state

    def material_mapping(self, state: DeclarationQAState) -> DeclarationQAState:
        """从退回信息中生成材料检索条件。"""
        next_state = dict(state)
        reject = next_state.get("reject_info") or {}
        if not reject:
            next_state["refuse_reason"] = "business_reject_not_found"
            add_event(next_state, "material_mapping_skipped", {"reason": "business_reject_not_found"})
            return next_state
        mapping = {
            "material_code": reject.get("material_code") or "",
            "material_name": reject.get("material_name") or "",
            "reject_code": reject.get("reject_code") or "",
            "audit_node": reject.get("audit_node") or "",
            "knowledge_tags": material_tags(reject),
        }
        next_state["material_mapping"] = mapping
        add_event(next_state, "material_mapping", mapping)
        return next_state

    def material_rule_rag(self, state: DeclarationQAState) -> DeclarationQAState:
        """检索补正材料规则知识。"""
        next_state = dict(state)
        reject = next_state.get("reject_info") or {}
        if not reject:
            next_state["retrieval_filter"] = {}
            next_state["retrieved_sources"] = []
            next_state["rag_status"] = {"skipped": True, "reason": "business_reject_not_found"}
            add_event(next_state, "rag_status", next_state["rag_status"])
            return next_state
        filters = base_policy_filter(next_state, allow_request_filter=False)
        query = " ".join(
            str(item)
            for item in [
                reject.get("reject_reason"),
                reject.get("material_name"),
                reject.get("reject_code"),
                next_state.get("normalized_query"),
            ]
            if item
        )
        result = self.rag_engine.retrieve(query, filters, top_k=4)
        next_state["retrieval_filter"] = filters
        next_state["retrieved_sources"] = [source.payload() for source in result.sources]
        next_state["rag_status"] = result.status
        add_event(next_state, "rag_status", result.status)
        return next_state
