"""Agent 工具注册、参数校验和统一执行边界。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ToolExecutionContext:
    """由服务端构造的可信工具执行上下文。"""

    user_id: int
    region_code: str = ""
    session_id: str = ""
    original_query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionResult:
    """工具的统一结构化结果。"""

    tool_name: str
    call_id: str
    ok: bool
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "ok": self.ok,
            "status": self.status,
            "data": dict(self.data),
            "error": self.error,
        }


class AgentTool(Protocol):
    """Registry 中工具必须满足的协议。"""

    name: str
    description: str
    input_model: type[BaseModel]

    def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolExecutionResult:
        """执行已经校验的参数。"""


class ToolRegistry:
    """按名称注册和执行 Agent 工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool, *, replace: bool = False) -> None:
        name = str(getattr(tool, "name", "") or "").strip()
        if not name:
            raise ValueError("tool name is required")
        if name in self._tools and not replace:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(str(name or "").strip())

    def definitions(self) -> list[dict[str, Any]]:
        """返回模型原生 Tool Calling 所需定义。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        context: ToolExecutionContext,
        *,
        call_id: str = "",
    ) -> ToolExecutionResult:
        tool_name = str(name or "").strip()
        invocation_id = str(call_id or tool_name or "tool_call")
        tool = self.get(tool_name)
        if tool is None:
            return ToolExecutionResult(tool_name, invocation_id, False, "tool_not_found", error=f"unregistered tool: {tool_name}")
        try:
            validated = tool.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolExecutionResult(tool_name, invocation_id, False, "invalid_arguments", error=str(exc)[:500])
        try:
            result = tool.execute(validated, context)
        except Exception as exc:
            return ToolExecutionResult(
                tool_name,
                invocation_id,
                False,
                "tool_error",
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        return ToolExecutionResult(
            tool_name=result.tool_name,
            call_id=invocation_id,
            ok=result.ok,
            status=result.status,
            data=result.data,
            error=result.error,
        )
