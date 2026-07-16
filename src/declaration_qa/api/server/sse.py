"""服务端事件响应序列化辅助函数。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from queue import Queue
from threading import Thread
from typing import Any

from declaration_qa.api.schemas import ChatResponse


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

def threaded_sse_event_stream(events: Iterable[dict]) -> Iterable[str]:
    """在固定后台线程中消费事件，再转为 SSE 数据流。"""
    queue: Queue[tuple[str, Any]] = Queue()

    def produce() -> None:
        try:
            for event in events:
                queue.put(("event", event))
        except Exception as exc:
            queue.put(("error", exc))
        finally:
            queue.put(("done", None))

    thread = Thread(target=produce, name="declaration-qa-sse-producer", daemon=True)
    thread.start()
    while True:
        kind, payload = queue.get()
        if kind == "event":
            yield sse_event(str(payload.get("event") or "event"), payload.get("data") or {})
            continue
        if kind == "error":
            yield sse_event("error", {"message": user_facing_error(payload)})
            break
        break


def sse_response(response: ChatResponse) -> Iterable[str]:
    """包装工作流事件为 SSE 响应。"""
    yield sse_event(
        "run_started",
        {"session_id": response.session_id, "audit_id": response.audit_id, "run_id": response.run_id},
    )
    for event in response.events:
        yield sse_event(str(event.get("event") or "event"), event.get("data") or {})
    if response.sources:
        yield sse_event("sources", {"items": [source.model_dump() for source in response.sources]})
    for chunk in chunk_text(response.answer):
        yield sse_event("token", {"content": chunk})
    yield sse_event(
        "done",
        {
            "session_id": response.session_id,
            "audit_id": response.audit_id,
            "run_id": response.run_id,
            "intent": response.intent,
            "route": response.route,
            "sub_intent": response.sub_intent,
            "agent_plan": response.agent_plan,
            "agent_query_plan": response.agent_query_plan,
            "retrieval_chain": response.retrieval_chain,
            "refused": response.refused,
            "refuse_reason": response.refuse_reason,
            "suggested_actions": [action.model_dump() for action in response.suggested_actions],
        },
    )


def user_facing_error(error: Exception) -> str:
    """把底层框架异常转换成适合前端展示的中文信息。"""
    if type(error).__name__ == "GraphRecursionError":
        return "Agent 执行步骤超过安全上限，请缩小查询范围后重试。"
    return str(error) or type(error).__name__


def sse_event(event: str, data: dict) -> str:
    """序列化单个 SSE 事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, *, size: int = 24) -> Iterable[str]:
    """按固定长度切分响应文本。"""
    value = str(text or "")
    if not value:
        return []
    return (value[index : index + size] for index in range(0, len(value), size))

