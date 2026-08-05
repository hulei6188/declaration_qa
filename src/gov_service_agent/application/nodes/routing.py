"""状态门禁与结构化路由节点。"""

from __future__ import annotations

from typing import Any

from gov_service_agent.application.helpers.events import add_event
from gov_service_agent.application.helpers.text_safety import is_unsafe_query
from gov_service_agent.application.routing import IntentRoute
from gov_service_agent.application.state import GovServiceAgentState
from gov_service_agent.application.state_machine import cancel_active_dialog_task
from gov_service_agent.application.tool_policy import allowed_tool_names


class RoutingNodes:
    """在 Agent 之前建立唯一、可审计的领域和流程边界。"""

    def state_gate(self, state: GovServiceAgentState) -> GovServiceAgentState:
        next_state = dict(state)
        task = dict(next_state.get("dialog_task") or {})
        if task.get("status") in {"completed", "cancelled"}:
            task = {}
            next_state["dialog_task"] = {}
        query = str(next_state.get("normalized_query") or "")
        if is_unsafe_query(query):
            next_state["refuse_reason"] = "unsafe_or_out_of_scope"
            gate = {"mode": "blocked", "reason": "unsafe_or_out_of_scope"}
        elif not query and not next_state.get("query_image") and not next_state.get("form_submission"):
            next_state["refuse_reason"] = "clarification_required"
            gate = {"mode": "blocked", "reason": "empty_request"}
        elif isinstance(next_state.get("form_submission"), dict):
            task_type = str(task.get("task_type") or "")
            domain = "complaint" if task_type == "complaint_create" else "application"
            gate = {"mode": "structured_form", "domain": domain, "reason": "trusted_form_submission"}
        else:
            gate = {"mode": "route", "reason": "model_routing_required"}
        next_state["workflow_gate"] = gate
        add_event(next_state, "state_gated", gate)
        return next_state

    def intent_route(self, state: GovServiceAgentState) -> GovServiceAgentState:
        next_state = dict(state)
        gate = dict(next_state.get("workflow_gate") or {})
        if gate.get("mode") == "blocked":
            decision = IntentRoute(
                domain="clarification",
                operation="clarify",
                needs_clarification=True,
                clarification_kind="question",
                clarification_question="请说明具体事项和希望办理或查询的内容。",
                reason=str(gate.get("reason") or "blocked"),
            ).model_dump()
            decision.update({"source": "state_gate", "router": "state_gate", "latency_ms": 0, "model": ""})
        elif gate.get("mode") == "structured_form":
            domain = str(gate.get("domain") or "application")
            decision = IntentRoute(
                domain=domain,
                operation="continue",
                item_name=str((next_state.get("dialog_task") or {}).get("item_name") or ""),
                requested_channel="online" if domain == "application" else "unspecified",
                confidence=1.0,
                reason="trusted_structured_form",
            ).model_dump()
            decision.update({"source": "state_gate", "router": "state_gate", "latency_ms": 0, "model": ""})
        else:
            decision = dict(self.intent_router.route(next_state))
        decision = normalize_route_decision(decision, next_state)
        task = dict(next_state.get("dialog_task") or {})
        if (
            task.get("status") in {"collecting", "ready"}
            and decision.get("operation") == "cancel"
            and decision.get("domain") in {"application", "complaint"}
        ):
            next_state["dialog_task"] = cancel_active_dialog_task(task)
            next_state["workflow_gate"] = {"mode": "dialog_cancelled", "reason": "explicit_user_cancel"}
        elif should_switch_dialog_domain(task, decision):
            next_state["dialog_task"] = cancel_active_dialog_task(task)
            next_state["workflow_gate"] = {
                "mode": "route",
                "reason": "explicit_domain_switch",
                "previous_task_type": str(task.get("task_type") or ""),
            }
        routed_domain = "clarification" if decision.get("needs_clarification") else str(decision.get("domain") or "clarification")
        next_state["route_decision"] = decision
        next_state["route"] = routed_domain
        next_state["intent"] = routed_domain
        next_state["sub_intent"] = str(decision.get("operation") or "clarify")
        next_state["routing_clarification_required"] = bool(decision.get("needs_clarification"))
        next_state["agent_tool_scope"] = sorted(allowed_tool_names(decision))
        add_event(
            next_state,
            "intent_routed",
            {
                "domain": routed_domain,
                "intended_domain": decision.get("domain"),
                "operation": decision.get("operation"),
                "item_name": decision.get("item_name") or "",
                "requested_channel": decision.get("requested_channel") or "unspecified",
                "confidence": float(decision.get("confidence") or 0.0),
                "needs_clarification": bool(decision.get("needs_clarification")),
                "clarification_kind": decision.get("clarification_kind") or "none",
                "tool_scope": list(next_state["agent_tool_scope"]),
                "router": decision.get("router") or "",
                "latency_ms": int(decision.get("latency_ms") or 0),
            },
        )
        return next_state


def normalize_route_decision(decision: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """在模型路由之后应用与业务事实无关的结构约束。"""
    normalized = IntentRoute.model_validate(
        {key: value for key, value in decision.items() if key in IntentRoute.model_fields}
    ).model_dump()
    normalized.update({key: decision.get(key) for key in ("source", "router", "latency_ms", "model")})
    domain = str(normalized.get("domain") or "")
    operation = str(normalized.get("operation") or "")
    item_name = str(normalized.get("item_name") or "").strip()
    if domain in {"application", "appointment"} and operation in {"start", "book"} and not item_name:
        normalized.update(
            {
                "needs_clarification": True,
                "clarification_kind": "item_name",
                "clarification_question": "请告诉我要办理的具体事项名称。",
            }
        )
    elif (
        domain in {"application", "appointment"}
        and operation == "start"
        and normalized.get("requested_channel") == "unspecified"
    ):
        normalized.update(
            {
                "needs_clarification": True,
                "clarification_kind": "service_channel",
                "clarification_question": "请选择线上申报或线下预约。",
            }
        )
    if normalized.get("needs_clarification") and not normalized.get("clarification_question"):
        normalized["clarification_question"] = "请补充具体事项和希望执行的操作。"
    return normalized


def should_switch_dialog_domain(task: dict[str, Any], decision: dict[str, Any]) -> bool:
    """显式启动另一事务领域时，确定性结束旧表单，避免两个写流程并存。"""
    if str(task.get("status") or "") not in {"collecting", "ready"}:
        return False
    current_domain = {
        "application_submission": "application",
        "complaint_create": "complaint",
    }.get(str(task.get("task_type") or ""), "")
    target_domain = str(decision.get("domain") or "")
    operation = str(decision.get("operation") or "")
    return bool(
        current_domain
        and target_domain in {"application", "appointment", "complaint"}
        and target_domain != current_domain
        and operation in {"start", "book", "create"}
        and not decision.get("needs_clarification")
    )
