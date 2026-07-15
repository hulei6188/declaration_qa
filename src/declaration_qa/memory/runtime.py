"""基于 LangGraph 持久化和 LangMem 的会话记忆运行时。"""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langmem import create_memory_store_manager
from langmem.short_term import SummarizationNode
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


logger = logging.getLogger(__name__)

LONG_TERM_MEMORY_INSTRUCTIONS = """只提取能够改善后续对话、由用户明确表达且长期稳定的记忆。

可以保存：稳定偏好、不包含敏感信息的稳定用户事实、用户明确提出的纠正，以及用户要求记住的可复用结论。

禁止保存：工单状态、申请进度、退回原因、政策正文、检索证据、工具结果、模型生成的回答、猜测、推断出的偏好、一次性问题、临时任务细节、身份号码、电话号码、地址、凭据或其他敏感个人数据。

每条记忆必须简洁、语义完整，并能追溯到用户的明确陈述。用户纠正已有信息时，应更新原记忆，不要创建重复记忆。"""


@dataclass
class MemoryResources:
    """封装已编译工作流和 LangMem 共享的持久化资源。"""

    checkpointer: BaseCheckpointSaver
    store: BaseStore
    pool: ConnectionPool | None = None

    def close(self) -> None:
        """关闭 PostgreSQL 连接池。"""
        if self.pool is not None:
            self.pool.close()
            self.pool = None


class LangGraphMemoryRuntime:
    """基于 LangGraph 持久化和 LangMem 的唯一记忆实现。"""

    def __init__(
        self,
        *,
        resources: MemoryResources,
        model: Any | None,
        summary_max_tokens: int,
        summary_trigger_tokens: int,
        summary_output_tokens: int,
        long_term_top_k: int,
        background_workers: int,
    ) -> None:
        """初始化会话摘要、长期记忆管理器和后台执行线程。"""
        self.resources = resources
        self.long_term_top_k = max(1, int(long_term_top_k))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(background_workers)),
            thread_name_prefix="langmem",
        )
        self._pending: set[Future[Any]] = set()
        self._closed = False
        self._summarizer = None
        self._manager = None
        if model is not None:
            summary_max = max(2, int(summary_max_tokens))
            summary_output = min(
                summary_max - 1,
                max(1, int(summary_output_tokens)),
            )
            summary_model = model.bind(max_tokens=summary_output)
            self._summarizer = SummarizationNode(
                model=summary_model,
                max_tokens=summary_max,
                max_tokens_before_summary=max(1, int(summary_trigger_tokens)),
                max_summary_tokens=summary_output,
                input_messages_key="messages",
                output_messages_key="summarized_messages",
            )
            self._manager = create_memory_store_manager(
                model,
                instructions=LONG_TERM_MEMORY_INSTRUCTIONS,
                namespace=("users", "{langgraph_user_id}", "memories"),
                store=resources.store,
                query_limit=self.long_term_top_k,
                enable_inserts=True,
                enable_deletes=False,
            )

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        """返回工作流使用的状态检查点存储器。"""
        return self.resources.checkpointer

    @property
    def store(self) -> BaseStore:
        """返回长期记忆使用的 LangGraph Store。"""
        return self.resources.store

    def summarize(
        self,
        messages: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> tuple[list[BaseMessage], dict[str, Any]]:
        """在消息超过阈值时生成滚动摘要，并返回本轮模型输入。"""
        converted = message_dicts_to_langchain(messages)
        if self._summarizer is None:
            return converted, dict(context or {})
        result = self._summarizer.invoke(
            {"messages": converted, "context": dict(context or {})}
        )
        return (
            list(result.get("summarized_messages") or converted),
            dict(result.get("context") or context or {}),
        )

    def retrieve(self, *, user_id: int, query: str) -> dict[str, Any]:
        """按用户命名空间检索与当前问题相关的长期记忆。"""
        if self._manager is None or not str(query or "").strip():
            return {"items": [], "count": 0, "backend": "langgraph_store"}
        items = self._manager.search(
            query=str(query),
            limit=self.long_term_top_k,
            config=memory_config(user_id),
        )
        normalized = [memory_search_item(item) for item in items]
        normalized = [item for item in normalized if item.get("content")]
        return {"items": normalized, "count": len(normalized), "backend": "langgraph_store"}

    def schedule_extraction(self, *, user_id: int, query: str, answer: str) -> dict[str, Any]:
        """在后台从一次成功问答中提取并更新长期记忆。"""
        if self._manager is None or not str(query or "").strip() or not str(answer or "").strip():
            return {"scheduled": False, "backend": "langmem"}
        future = self._executor.submit(
            self._manager.invoke,
            {
                "messages": [
                    HumanMessage(content=str(query)),
                    AIMessage(content=str(answer)),
                ]
            },
            memory_config(user_id),
        )
        self._pending.add(future)
        future.add_done_callback(self._extraction_done)
        return {"scheduled": True, "backend": "langmem"}

    def _extraction_done(self, future: Future[Any]) -> None:
        """回收后台任务并记录长期记忆提取异常。"""
        self._pending.discard(future)
        try:
            future.result()
        except Exception:
            logger.exception("LangMem 后台长期记忆提取失败")

    def close(self) -> None:
        """等待后台任务结束并释放全部记忆资源。"""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.resources.close()


def build_memory_resources(
    *,
    database_url: str | None,
    embedding_provider: Any,
    pool_min_size: int = 1,
    pool_max_size: int = 10,
) -> MemoryResources:
    """创建工作流状态与长期记忆共用的唯一持久化实现。"""
    index = {
        "dims": int(embedding_provider.dimension),
        "embed": embedding_provider.embed_documents,
    }
    if not database_url:
        return MemoryResources(
            checkpointer=InMemorySaver(),
            store=InMemoryStore(index=index),
        )
    pool = ConnectionPool(
        conninfo=psycopg_connection_url(database_url),
        min_size=max(1, int(pool_min_size)),
        max_size=max(1, int(pool_max_size)),
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=True,
    )
    try:
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool, index=index)
        checkpointer.setup()
        store.setup()
    except Exception:
        pool.close()
        raise
    return MemoryResources(checkpointer=checkpointer, store=store, pool=pool)


def message_dicts_to_langchain(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """将项目消息字典转换为 LangChain 标准消息对象。"""
    output: list[BaseMessage] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("id") or uuid4().hex)
        role = str(item.get("role") or "")
        if role == "user":
            output.append(
                HumanMessage(content=str(item.get("content") or ""), id=message_id)
            )
        elif role == "assistant":
            output.append(
                AIMessage(
                    content=str(item.get("content") or ""),
                    tool_calls=list(item.get("tool_calls") or []),
                    id=message_id,
                )
            )
        elif role == "tool" and item.get("tool_call_id"):
            content = item.get("content")
            output.append(
                ToolMessage(
                    content=(
                        content
                        if isinstance(content, str)
                        else json.dumps(content or {}, ensure_ascii=False, default=str)
                    ),
                    tool_call_id=str(item["tool_call_id"]),
                    name=str(item.get("name") or ""),
                    id=message_id,
                )
            )
    return output


def memory_search_item(item: Any) -> dict[str, Any]:
    """将 LangMem 检索结果整理为可注入 Agent 上下文的结构。"""
    value = getattr(item, "value", None) or {}
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    content = value.get("content") if isinstance(value, dict) else value
    if hasattr(content, "model_dump"):
        content = content.model_dump()
    if isinstance(content, dict):
        content = (
            content.get("content")
            or content.get("text")
            or json.dumps(content, ensure_ascii=False)
        )
    return {
        "key": str(getattr(item, "key", "") or ""),
        "content": str(content or "").strip(),
        "score": getattr(item, "score", None),
    }


def memory_config(user_id: int) -> dict[str, Any]:
    """构造按用户隔离长期记忆的 LangGraph 配置。"""
    return {"configurable": {"langgraph_user_id": str(int(user_id))}}


def psycopg_connection_url(database_url: str) -> str:
    """将 SQLAlchemy PostgreSQL URL 转换为 Psycopg 可识别的 URL。"""
    return str(database_url).replace("postgresql+psycopg://", "postgresql://", 1)
