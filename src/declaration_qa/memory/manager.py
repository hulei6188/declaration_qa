"""上下文记忆管理器。"""

from __future__ import annotations

import hashlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from declaration_qa.memory.extraction import ContextMemoryExtractor
from declaration_qa.memory.models import ContextMemoryItem, context_memory_from_payload
from declaration_qa.memory.retrieval import ContextMemoryRetriever
from declaration_qa.memory.store import ContextMemoryStore
from declaration_qa.observability.langfuse import observation, record_error, safe_value
from declaration_qa.memory.context import RECENT_TURN_LIMIT
from declaration_qa.storage.cache import CacheStore


SHORT_MEMORY_VERSION = 4
LONG_MEMORY_VERSION = 1


class ContextMemoryManager:
    """分层上下文记忆管理器。"""

    def __init__(
        self,
        *,
        cache_store: CacheStore,
        memory_store: ContextMemoryStore,
        retriever: ContextMemoryRetriever,
        short_ttl_seconds: int = 3600,
        long_ttl_seconds: int = 604800,
        retrieval_cache_ttl_seconds: int = 300,
        max_long_memories_per_user: int = 200,
        long_memory_top_k: int = 5,
        async_workers: int = 2,
        enabled: bool = True,
    ) -> None:
        """初始化分层记忆管理器。"""
        self.cache_store = cache_store
        self.memory_store = memory_store
        self.retriever = retriever
        self.extractor = ContextMemoryExtractor()
        self.short_ttl_seconds = max(1, int(short_ttl_seconds or 3600))
        self.long_ttl_seconds = max(1, int(long_ttl_seconds or 604800))
        self.retrieval_cache_ttl_seconds = max(1, int(retrieval_cache_ttl_seconds or 300))
        self.max_long_memories_per_user = max(10, int(max_long_memories_per_user or 200))
        self.long_memory_top_k = max(1, int(long_memory_top_k or 5))
        self.enabled = bool(enabled)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(async_workers or 1)), thread_name_prefix="context-memory")
        self._pending: list[Future] = []

    def load_short_memory(
        self,
        *,
        user_id: int,
        session_id: str,
    ) -> dict[str, Any]:
        """读取 Redis 短期上下文记忆；旧版本缓存只迁移 Agent 消息。"""
        if not self.enabled or not session_id:
            return build_short_memory_payload(cache_hit=False, backend="disabled")
        key = short_memory_key(user_id=user_id, session_id=session_id)
        with observation("memory.short.load", input={"user_id": user_id, "session_id": session_id}) as span:
            cached = self.cache_store.get_json(key)
            if cached:
                short_memory = build_short_memory_payload(
                    agent_messages=cached.get("agent_messages") or [],
                    cache_hit=True,
                    backend=self.cache_store.backend,
                )
                if cached.get("version") != SHORT_MEMORY_VERSION:
                    self.cache_store.set_json(key, short_memory, ttl_seconds=self.short_ttl_seconds)
                span.update(output={"hit": True, "backend": self.cache_store.backend})
                return short_memory
            payload = build_short_memory_payload(cache_hit=False, backend=self.cache_store.backend)
            self.cache_store.set_json(key, payload, ttl_seconds=self.short_ttl_seconds)
            span.update(output={"hit": False, "backend": self.cache_store.backend, "agent_message_count": 0})
            return payload

    def update_short_memory(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        """每轮交互结束后刷新 Redis 短期记忆。"""
        if not self.enabled:
            return {}
        user_id = int(payload.get("user_id") or 0)
        session_id = str(payload.get("session_id") or "")
        if not user_id or not session_id:
            return {}
        existing = self.cache_store.get_json(short_memory_key(user_id=user_id, session_id=session_id)) or {}
        short_memory = build_short_memory_payload(
            cache_hit=False,
            backend=self.cache_store.backend,
            agent_messages=payload.get("agent_messages") or existing.get("agent_messages") or [],
        )
        self.cache_store.set_json(short_memory_key(user_id=user_id, session_id=session_id), short_memory, ttl_seconds=self.short_ttl_seconds)
        return short_memory

    def retrieve_long_term_context(self, state: dict[str, Any]) -> dict[str, Any]:
        """检索长期上下文记忆，结果同时进入 Redis 检索缓存。"""
        if not self.enabled:
            return empty_long_context(reason="disabled")
        user_id = int(state.get("user_id") or 0)
        query = long_context_query(state)
        if not user_id or not query:
            return empty_long_context(reason="empty_query")
        cache_key = long_retrieval_cache_key(user_id=user_id, query=query)
        with observation("memory.long.retrieve", input={"user_id": user_id, "query": query}) as span:
            cached = self.cache_store.get_json(cache_key)
            if cached:
                cached["cache"] = {"hit": True, "backend": self.cache_store.backend}
                span.update(output=safe_value({"cache_hit": True, "count": len(cached.get("items") or [])}))
                return cached
            try:
                hits, status = self.retriever.retrieve(query, user_id=user_id, limit=int(state.get("memory_top_k") or self.long_memory_top_k))
            except Exception as exc:
                record_error(span, exc, stage="memory.long.retrieve", reason=type(exc).__name__, extra={"user_id": user_id})
                context = self._fallback_long_context_from_redis(user_id=user_id, reason=type(exc).__name__)
                span.update(output=safe_value(context))
                return context
            context = {
                "version": LONG_MEMORY_VERSION,
                "query": query,
                "items": [hit.payload() for hit in hits],
                "status": status,
                "cache": {"hit": False, "backend": self.cache_store.backend},
            }
            if context["items"]:
                self.cache_store.set_json(cache_key, context, ttl_seconds=self.retrieval_cache_ttl_seconds)
            span.update(output=safe_value({"cache_hit": False, "count": len(context["items"]), "status": status}))
            return context

    def after_interaction(self, payload: dict[str, Any], *, audit_id: str) -> dict[str, Any]:
        """每轮交互结束后的同步短期更新和异步长期记忆任务。"""
        if not self.enabled:
            return {"enabled": False}
        short_memory = self.update_short_memory(payload=payload)
        future = self._executor.submit(self._index_long_term_memories, dict(payload), audit_id)
        self._pending.append(future)
        return {
            "enabled": True,
            "short_memory": {
                "backend": self.cache_store.backend,
                "agent_message_count": len(short_memory.get("agent_messages") or []),
            },
            "long_memory": {"async_enqueued": True},
        }

    def wait_for_pending(self, timeout_seconds: float = 10.0) -> list[Any]:
        """测试和管理场景等待已提交异步任务完成。"""
        pending = list(self._pending)
        self._pending.clear()
        results = []
        for future in pending:
            results.append(future.result(timeout=timeout_seconds))
        return results

    def _index_long_term_memories(self, payload: dict[str, Any], audit_id: str) -> dict[str, Any]:
        """后台线程抽取、持久化并索引长期记忆。"""
        with observation("memory.long.index", input={"audit_id": audit_id, "user_id": payload.get("user_id")}) as span:
            try:
                memories = self.extractor.extract(payload, audit_id=audit_id)
                for memory in memories:
                    self.memory_store.upsert_memory(memory)
                self.retriever.index_memories(memories)
                self._cache_long_memories(memories)
                output = {"memory_count": len(memories), "ids": [memory.id for memory in memories]}
                span.update(output=safe_value(output))
                return output
            except Exception as exc:
                record_error(span, exc, stage="memory.long.index", reason=type(exc).__name__, extra={"audit_id": audit_id})
                span.update(output={"error": type(exc).__name__})
                return {"memory_count": 0, "error": type(exc).__name__}

    def _cache_long_memories(self, memories: list[ContextMemoryItem]) -> None:
        """把长期记忆副本写入 Redis，作为热点缓存和降级检索源。"""
        by_user: dict[int, list[ContextMemoryItem]] = {}
        for memory in memories:
            by_user.setdefault(memory.user_id, []).append(memory)
        for user_id, user_memories in by_user.items():
            key = long_memory_list_key(user_id=user_id)
            cached = self.cache_store.get_json(key) or {"items": []}
            items = [context_memory_from_payload(item) for item in cached.get("items") or [] if isinstance(item, dict)]
            merged = {item.id: item for item in items}
            for memory in user_memories:
                merged[memory.id] = memory
            ordered = sorted(merged.values(), key=lambda item: item.created_at, reverse=True)[: self.max_long_memories_per_user]
            self.cache_store.set_json(key, {"version": LONG_MEMORY_VERSION, "items": [item.payload() for item in ordered]}, ttl_seconds=self.long_ttl_seconds)

    def _fallback_long_context_from_redis(self, *, user_id: int, reason: str) -> dict[str, Any]:
        """外部索引异常时从 Redis 长期记忆列表降级读取。"""
        cached = self.cache_store.get_json(long_memory_list_key(user_id=user_id)) or {}
        items = list(cached.get("items") or [])[:5]
        return {
            "version": LONG_MEMORY_VERSION,
            "query": "",
            "items": items,
            "status": {"retrieval_mode": "redis_fallback", "reason": reason},
            "cache": {"hit": True, "backend": self.cache_store.backend},
        }


def build_short_memory_payload(
    *,
    cache_hit: bool,
    backend: str,
    agent_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建短期记忆缓存载荷。"""
    return {
        "version": SHORT_MEMORY_VERSION,
        "agent_messages": normalize_agent_messages(agent_messages or []),
        "cache_hit": cache_hit,
        "backend": backend,
    }


def normalize_agent_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留最近完整轮次的可重放消息，不截断工具结果。"""
    output: list[dict[str, Any]] = []
    for message in list(messages or []):
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "tool"}:
            continue
        item: dict[str, Any] = {
            "role": str(message.get("role") or ""),
            "content": message.get("content") if isinstance(message.get("content"), (dict, list, str)) else str(message.get("content") or ""),
        }
        if message.get("tool_call_id"):
            item["tool_call_id"] = str(message["tool_call_id"])
        if message.get("name"):
            item["name"] = str(message["name"])
        if message.get("tool_calls"):
            item["tool_calls"] = list(message["tool_calls"])
        output.append(item)
    user_indexes = [index for index, item in enumerate(output) if item.get("role") == "user"]
    if len(user_indexes) > RECENT_TURN_LIMIT:
        return output[user_indexes[-RECENT_TURN_LIMIT] :]
    return output


def long_context_query(state: dict[str, Any]) -> str:
    """构造长期记忆检索 query。"""
    candidates = [
        state.get("retrieval_query"),
        state.get("normalized_query"),
        state.get("query"),
        state.get("rag_query"),
    ]
    return " ".join(str(item or "").strip() for item in candidates if str(item or "").strip())[:800]


def empty_long_context(*, reason: str) -> dict[str, Any]:
    """构造空长期记忆上下文。"""
    return {
        "version": LONG_MEMORY_VERSION,
        "items": [],
        "status": {"retrieval_mode": "none", "reason": reason},
        "cache": {"hit": False},
    }


def short_memory_key(*, user_id: int, session_id: str) -> str:
    """Redis 短期记忆 key。"""
    return f"declaration_qa:context:short:{int(user_id)}:{session_id}"


def long_memory_list_key(*, user_id: int) -> str:
    """Redis 用户长期记忆列表 key。"""
    return f"declaration_qa:context:long:list:{int(user_id)}"


def long_retrieval_cache_key(*, user_id: int, query: str) -> str:
    """Redis 长期记忆检索结果缓存 key。"""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]
    return f"declaration_qa:context:long:retrieve:{int(user_id)}:{digest}"
