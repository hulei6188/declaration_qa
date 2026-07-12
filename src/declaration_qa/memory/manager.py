"""上下文记忆管理器。"""

from __future__ import annotations

import hashlib
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

from declaration_qa.memory.extraction import ContextMemoryExtractor
from declaration_qa.memory.models import ContextMemoryItem, context_memory_from_payload
from declaration_qa.memory.retrieval import ContextMemoryRetriever
from declaration_qa.memory.store import ContextMemoryStore
from declaration_qa.observability.langfuse import observation, record_error, safe_value
from declaration_qa.memory.context import normalize_recent_messages
from declaration_qa.storage.cache import CacheStore


SHORT_MEMORY_VERSION = 3
LONG_MEMORY_VERSION = 1
SHORT_MEMORY_ITEM_LIMIT = 12


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
        verified_ticket_facts = merge_verified_ticket_facts(
            existing.get("verified_ticket_facts") or existing.get("verified_facts") or [],
            extract_verified_ticket_facts(payload.get("tool_results") or []),
        )
        ticket_item_overviews = merge_ticket_item_overviews(
            existing.get("ticket_item_overviews") or [],
            extract_ticket_item_overviews(payload.get("tool_results") or []),
        )
        recent_knowledge_evidence_refs = merge_knowledge_evidence_refs(
            existing.get("recent_knowledge_evidence_refs") or [],
            extract_knowledge_evidence_refs(
                payload.get("retrieved_sources") or [],
                session_id=session_id,
                query=str(payload.get("query") or ""),
                ttl_seconds=self.short_ttl_seconds,
            ),
        )
        short_memory = build_short_memory_payload(
            messages,
            cache_hit=False,
            backend=self.cache_store.backend,
            verified_ticket_facts=verified_ticket_facts,
            ticket_item_overviews=ticket_item_overviews,
            recent_knowledge_evidence_refs=recent_knowledge_evidence_refs,
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
                "recent_message_count": len(short_memory.get("recent_messages") or []),
                "summary": short_memory.get("summary") or "",
                "verified_ticket_fact_count": len(short_memory.get("verified_ticket_facts") or []),
                "ticket_item_overview_count": len(short_memory.get("ticket_item_overviews") or []),
                "knowledge_evidence_ref_count": len(short_memory.get("recent_knowledge_evidence_refs") or []),
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
    messages: list[dict[str, Any]],
    *,
    cache_hit: bool,
    backend: str,
    verified_ticket_facts: list[dict[str, Any]] | None = None,
    ticket_item_overviews: list[dict[str, Any]] | None = None,
    recent_knowledge_evidence_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建短期记忆缓存载荷。"""
    recent_messages = normalize_recent_messages(messages)
    return {
        "version": SHORT_MEMORY_VERSION,
        "summary": summarize_recent_messages(recent_messages),
        "recent_messages": recent_messages,
        "recent_message_count": len(recent_messages),
        "verified_ticket_facts": list(verified_ticket_facts or [])[-SHORT_MEMORY_ITEM_LIMIT:],
        "ticket_item_overviews": list(ticket_item_overviews or [])[-SHORT_MEMORY_ITEM_LIMIT:],
        "recent_knowledge_evidence_refs": list(recent_knowledge_evidence_refs or [])[-SHORT_MEMORY_ITEM_LIMIT:],
        "cache_hit": cache_hit,
        "backend": backend,
    }


def extract_verified_ticket_facts(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从成功的工单详情结果提取可追溯、可过期的短期事实。"""
    observed_at = datetime.now(timezone.utc).isoformat()
    facts: list[dict[str, Any]] = []
    for result in tool_results:
        if not isinstance(result, dict) or not result.get("ok") or result.get("tool_name") != "ticket_detail_query":
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        item_code = str(data.get("item_code") or "")
        for application in data.get("applications") or []:
            if not isinstance(application, dict):
                continue
            history = [item for item in (application.get("status_history") or []) if isinstance(item, dict)]
            reason_event = next(
                (item for item in reversed(history) if item.get("reason") or item.get("material_name")),
                {},
            )
            application_id = int(application.get("application_id") or 0)
            if not item_code or not application_id:
                continue
            facts.append(
                {
                    "source": "ticket_detail_query",
                    "fact_type": "application_status",
                    "trusted": True,
                    "item_code": item_code,
                    "item_name": str(application.get("item_name") or ""),
                    "application_id": application_id,
                    "current_status": str(application.get("current_status") or ""),
                    "reason": str(reason_event.get("reason") or ""),
                    "material_name": str(reason_event.get("material_name") or ""),
                    "fact_updated_at": str(application.get("updated_at") or reason_event.get("changed_at") or ""),
                    "verified_at": observed_at,
                    "refresh_required_for_current_status": True,
                }
            )
    return facts


def merge_verified_ticket_facts(
    existing: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按事项、申请和事实类型覆盖旧事实，限制短期缓存体积。"""
    merged: dict[tuple[str, int, str], dict[str, Any]] = {}
    for fact in [*existing, *current]:
        if not isinstance(fact, dict) or not fact.get("trusted"):
            continue
        key = (
            str(fact.get("item_code") or ""),
            int(fact.get("application_id") or 0),
            str(fact.get("fact_type") or ""),
        )
        if key[0] and key[1] and key[2]:
            merged[key] = dict(fact)
    return list(merged.values())[-SHORT_MEMORY_ITEM_LIMIT:]


def extract_ticket_item_overviews(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从成功的工单列表结果提取事项级短期概览。"""
    verified_at = datetime.now(timezone.utc).isoformat()
    output: list[dict[str, Any]] = []
    for result in tool_results:
        if not isinstance(result, dict) or not result.get("ok") or result.get("tool_name") != "ticket_list_query":
            continue
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        for item in data.get("items") or []:
            if not isinstance(item, dict) or not str(item.get("item_code") or "").strip():
                continue
            output.append(
                {
                    "source": "ticket_list_query",
                    "item_code": str(item.get("item_code") or ""),
                    "item_name": str(item.get("item_name") or ""),
                    "application_count": int(item.get("application_count") or 0),
                    "statuses": [str(value) for value in (item.get("statuses") or []) if str(value)],
                    "fact_updated_at": str(item.get("updated_at") or ""),
                    "verified_at": verified_at,
                    "refresh_required": True,
                }
            )
    return output


def merge_ticket_item_overviews(
    existing: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """同一事项保留最近一次列表概览。"""
    merged: dict[str, dict[str, Any]] = {}
    for item in [*existing, *current]:
        if not isinstance(item, dict) or item.get("source") != "ticket_list_query":
            continue
        item_code = str(item.get("item_code") or "")
        if item_code:
            merged[item_code] = dict(item)
    return list(merged.values())[-SHORT_MEMORY_ITEM_LIMIT:]


def extract_knowledge_evidence_refs(
    sources: list[dict[str, Any]],
    *,
    session_id: str,
    query: str,
    ttl_seconds: int,
) -> list[dict[str, Any]]:
    """把公开知识来源压缩为当前会话可校验的短期证据引用。"""
    retrieved_at = datetime.now(timezone.utc)
    expires_at = retrieved_at + timedelta(seconds=max(1, int(ttl_seconds or 1)))
    output: list[dict[str, Any]] = []
    for source in sources[:SHORT_MEMORY_ITEM_LIMIT]:
        if not isinstance(source, dict):
            continue
        chunk_id = str(source.get("chunk_id") or "").strip()
        doc_id = str(source.get("doc_id") or "").strip()
        if not chunk_id or not doc_id:
            continue
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        digest = hashlib.sha256(
            f"{session_id}:{chunk_id}:{retrieved_at.isoformat()}".encode("utf-8")
        ).hexdigest()[:32]
        evidence_text = " ".join(
            str(source.get("evidence_text") or source.get("snippet") or "").split()
        )
        output.append(
            {
                "evidence_ref": f"ev_{digest}",
                "session_id": session_id,
                "query": " ".join(query.split())[:300],
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": str(source.get("title") or "")[:200],
                "section": str(metadata.get("section_path") or metadata.get("node_title") or "")[:300],
                "summary": evidence_text[:320],
                "region_code": str(source.get("region_code") or ""),
                "policy_version": str(source.get("policy_version") or ""),
                "status": str(metadata.get("status") or metadata.get("doc_status") or "published"),
                "effective_start_time": source.get("effective_start_time"),
                "effective_end_time": source.get("effective_end_time"),
                "retrieved_at": retrieved_at.isoformat(),
                "expires_at": expires_at.isoformat(),
            }
        )
    return output


def merge_knowledge_evidence_refs(
    existing: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """移除过期引用并按 chunk 覆盖为最近一次召回记录。"""
    now = datetime.now(timezone.utc)
    merged: dict[str, dict[str, Any]] = {}
    for item in [*existing, *current]:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        expires_at = parse_utc_datetime(item.get("expires_at"))
        if not chunk_id or not expires_at or expires_at <= now:
            continue
        merged[chunk_id] = dict(item)
    return list(merged.values())[-SHORT_MEMORY_ITEM_LIMIT:]


def parse_utc_datetime(value: Any) -> datetime | None:
    """解析带时区时间；无时区值按 UTC 处理。"""
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
