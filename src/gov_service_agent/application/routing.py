"""结构化意图路由器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from gov_service_agent.observability.langfuse import invoke_with_langfuse


RouteDomain = Literal[
    "knowledge",
    "business_query",
    "application",
    "appointment",
    "complaint",
    "conversation",
    "clarification",
]
RouteOperation = Literal[
    "consult",
    "list",
    "detail",
    "status",
    "queue",
    "payment",
    "receipt",
    "result",
    "delivery",
    "start",
    "continue",
    "book",
    "cancel",
    "create",
    "chat",
    "clarify",
]


class IntentRoute(BaseModel):
    """一级领域路由的稳定输出契约。"""

    model_config = ConfigDict(extra="forbid")

    domain: RouteDomain
    operation: RouteOperation
    item_name: str = Field(default="", max_length=160)
    requested_channel: Literal["online", "offline", "unspecified"] = "unspecified"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_kind: Literal["none", "item_name", "service_channel", "target", "question"] = "none"
    clarification_question: str = Field(default="", max_length=240)
    reason: str = Field(default="", max_length=240)


class IntentRouter(Protocol):
    """工作流使用的路由器协议。"""

    name: str

    def route(self, state: dict[str, Any]) -> dict[str, Any]:
        """返回可写入工作流状态的结构化路由结果。"""


ROUTER_SYSTEM_PROMPT = """你是政务服务工作流的一级路由器，只分类，不回答问题，也不调用业务工具。

必须输出 IntentRoute：
- knowledge：公开政策、条件、材料、流程、费用、时限、样表或法规咨询。
- business_query：个人已有办件、进度、状态、退回、缴费、回执、证照、送达或排队查询。
- application：明确开始或继续线上申报。
- appointment：查询预约、号源、大厅，或明确预约、取消预约、线下办理。
- complaint：创建投诉，或查询投诉/咨询记录。
- conversation：寒暄、确认上一轮回答、模型身份、语言解释等无需外部事实的对话。
- clarification：输入无法形成可靠任务。

约束：
1. “需要什么材料/怎么办/条件是什么”属于 knowledge，即使包含“办理”。
2. “我想办理一个”缺少事项名称：needs_clarification=true，kind=item_name；不得先询问渠道。
3. 明确要办理且事项已知、但未说明线上申报或线下预约：needs_clarification=true，kind=service_channel。
4. 用户明确线上/网上/在线申报时 domain=application；明确线下/窗口/现场/预约时 domain=appointment。
5. 有 collecting 表单时，普通咨询仍按咨询内容分类；只有明确继续填报、修改表单或结构化表单提交才属于 continue。
6. “真的假的/确定吗/你刚才说的是真的吗”通常是 conversation，不能据此启动或继续写流程。
7. 不得猜测事项名称、业务主键或用户未表达的渠道。
8. confidence 反映分类把握；需要澄清时必须给出一句简短 clarification_question。
"""


class StructuredIntentRouter:
    """通过模型原生结构化 Tool Call 执行一级路由。"""

    name = "structured_model_router"

    def __init__(self, model: Any | None) -> None:
        self.model = model

    def route(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            return router_unavailable_decision("router_model_unavailable")
        from langchain_core.messages import HumanMessage, SystemMessage

        recent_messages = []
        for message in reversed(state.get("agent_messages") or []):
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                continue
            content = str(message.get("content") or "").strip()
            if content:
                recent_messages.append({"role": message["role"], "content": content[:500]})
            if len(recent_messages) >= 6:
                break
        recent_messages.reverse()
        payload = {
            "query": str(state.get("normalized_query") or state.get("query") or ""),
            "has_query_image": bool(state.get("query_image")),
            "query_image_ocr_text": str(state.get("query_image_ocr_text") or "")[:2000],
            "active_dialog_task": compact_dialog_task(state.get("dialog_task")),
            "recent_conversation": recent_messages,
        }
        started_at = perf_counter()
        try:
            bound = self.model.bind_tools([IntentRoute], tool_choice="IntentRoute")
            response = invoke_with_langfuse(
                bound,
                [
                    SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
                ],
                state,
                run_name="intent.route",
            )
            calls = list(getattr(response, "tool_calls", None) or [])
            if not calls:
                return router_unavailable_decision("router_missing_structured_output")
            decision = IntentRoute.model_validate(dict(calls[0].get("args") or {})).model_dump()
            decision.update(
                {
                    "source": "model",
                    "router": self.name,
                    "latency_ms": max(0, round((perf_counter() - started_at) * 1000)),
                    "model": str(
                        (getattr(response, "response_metadata", None) or {}).get("model_name")
                        or getattr(self.model, "model_name", "")
                        or getattr(self.model, "model", "")
                    ),
                }
            )
            return decision
        except Exception as exc:
            return router_unavailable_decision(f"router_error:{type(exc).__name__}")


@dataclass
class StaticIntentRouter:
    """测试和受控调用使用的静态结构化路由器。"""

    decision: dict[str, Any]
    name: str = "static_router"

    def route(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            **IntentRoute.model_validate(self.decision).model_dump(),
            "source": "static",
            "router": self.name,
            "latency_ms": 0,
            "model": "",
        }


def router_unavailable_decision(reason: str) -> dict[str, Any]:
    """路由模型不可用时安全停在澄清态，不向 Agent 暴露工具。"""
    return {
        **IntentRoute(
            domain="clarification",
            operation="clarify",
            confidence=0.0,
            needs_clarification=True,
            clarification_kind="question",
            clarification_question="我暂时无法判断你要办理的业务类型，请说明具体事项和希望执行的操作。",
            reason=reason,
        ).model_dump(),
        "source": "guardrail",
        "router": "safe_fallback",
        "latency_ms": 0,
        "model": "",
    }


def compact_dialog_task(value: Any) -> dict[str, Any]:
    """只向路由器暴露流程判断所需的最小表单状态。"""
    task = value if isinstance(value, dict) else {}
    next_field = task.get("next_field") if isinstance(task.get("next_field"), dict) else {}
    return {
        "task_type": str(task.get("task_type") or ""),
        "status": str(task.get("status") or ""),
        "item_code": str(task.get("item_code") or ""),
        "item_name": str(task.get("item_name") or ""),
        "next_field": str(next_field.get("name") or ""),
    }
