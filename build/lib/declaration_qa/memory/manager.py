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
from declaration_qa.memory.context import normalize_recent_messages
from declaration_qa.storage.cache import CacheStore


SHORT_MEMORY_VERSION = 1
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
        fallback_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """读取 Redis 短期上下文记忆；未命中时使用会话消息构建并回写。"""
        if not self.enabled or not session_id:
            return build_short_memory_payload(fallback_messages, cache_hit=False, backend="disabled")
        key = short_memory_key(user_id=user_id, session_id=session_id)
        with observation("memory.short.load", input={"user_id": user_id, "session_id": session_id}) as span:
            cached = self.cache_store.get_json(key)
            if cached:
                cached["cache_hit"] = True
                cached["backend"] = self.cache_store.backend
                span.update(output={"hit": True, "backend": self.cache_store.backend})
                return cached
            payload = build_short_memory_payload(fallback_messages, cache_hit=False, backend=self.cache_store.backend)
            self.cache_store.set_json(key, payload, ttl_seconds=self.short_ttl_seconds)
            span.update(output={"hit": False, "backend": self.cache_store.backend, "message_count": len(payload.get("recent_messages") or [])})
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
        messages = list(existing.get("recent_messages") or [])
        messages.extend(
            [
                {
                    "role": "user",
                    "content": payload.get("query") or "",
                    "attachments": query_image_attachments(payload.get("query_image")),
                },
                {
                    "role": "assistant",
                    "content": payload.get("answer") or "",
                    "intent": payload.get("intent") or "",
                    "refused": bool(payload.get("refused")),
                },
            ]
        )
        short_memory = build_short_memory_payload(messages, cache_hit=False, backend=self.cache_store.backend)
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
                "recent_message_count": len(short_memory.get("recent_messages") or []),
                "summary": short_memory.get("summary") or "",
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


def build_short_memory_payload(messages: list[dict[str, Any]], *, cache_hit: bool, backend: str) -> dict[str, Any]:
    """构建短期记忆缓存载荷。"""
    recent_messages = normalize_recent_messages(messages)
    return {
        "version": SHORT_MEMORY_VERSION,
        "summary": summarize_recent_messages(recent_messages),
        "recent_messages": recent_messages,
        "recent_message_count": len(recent_messages),
        "cache_hit": cache_hit,
        "backend": backend,
    }


def summarize_recent_messages(messages: list[dict[str, Any]]) -> str:
    """用抽取式方式生成短期会话摘要。"""
    pairs = []
    for message in messages[-6:]:
        role = "用户" if message.get("role") == "user" else "助手"
        content = " ".join(str(message.get("content") or "").split())
        if content:
            pairs.append(f"{role}：{content[:160]}")
    return "\n".join(pairs)[-1200:]


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


def query_image_attachments(query_image: Any) -> list[dict[str, Any]]:
    """把当前图片摘要转为短期消息附件。"""
    if not isinstance(query_image, dict) or not query_image:
        return []
    filename = str(query_image.get("filename") or "").strip()
    if not filename:
        return []
    return [
        {
            "filename": filename,
            "content_type": str(query_image.get("content_type") or "image/png"),
            "size_bytes": int(query_image.get("size_bytes") or 0),
        }
    ]


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
