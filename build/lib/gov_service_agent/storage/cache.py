"""缓存存储协议以及内存和 Redis 实现。"""

from __future__ import annotations

import json
from typing import Any, Protocol


class CacheStore(Protocol):
    """JSON 缓存读写能力的统一协议。"""
    backend: str

    def get_json(self, key: str) -> dict[str, Any] | None:
        """读取指定 key 的 JSON 缓存值。"""
        ...

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        """写入指定 key 的 JSON 缓存值。"""
        ...

    def clear(self) -> None:
        """清空当前存储中的全部数据。"""
        ...


class InMemoryCacheStore:
    """进程内 JSON 缓存实现。"""
    backend = "memory"

    def __init__(self) -> None:
        """初始化内存缓存。"""
        self._items: dict[str, dict[str, Any]] = {}

    def get_json(self, key: str) -> dict[str, Any] | None:
        """读取指定 key 的 JSON 缓存值。"""
        item = self._items.get(key)
        return dict(item) if item else None

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        """写入指定 key 的 JSON 缓存值。"""
        self._items[key] = dict(value)

    def clear(self) -> None:
        """清空当前存储中的全部数据。"""
        self._items.clear()


class RedisCacheStore:
    """基于 Redis 的 JSON 缓存实现。"""
    backend = "redis"
    key_registry = "gov_service_agent:cache_keys"

    def __init__(self, redis_url: str, *, allow_fallback: bool = True) -> None:
        """初始化Redis 缓存。"""
        import redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=1)
        self._available = True
        self._fallback = InMemoryCacheStore()
        try:
            self._client.ping()
        except Exception:
            if not allow_fallback:
                raise
            self._available = False

    def get_json(self, key: str) -> dict[str, Any] | None:
        """读取指定 key 的 JSON 缓存值。"""
        if not self._available:
            return self._fallback.get_json(key)
        raw = self._client.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        """写入指定 key 的 JSON 缓存值。"""
        if not self._available:
            self._fallback.set_json(key, value, ttl_seconds=ttl_seconds)
            return
        self._client.setex(key, max(1, int(ttl_seconds)), json.dumps(value, ensure_ascii=False, default=str))
        self._client.sadd(self.key_registry, key)

    def clear(self) -> None:
        """清空当前存储中的全部数据。"""
        if not self._available:
            self._fallback.clear()
            return
        keys = [str(item) for item in self._client.smembers(self.key_registry)]
        if keys:
            self._client.delete(*keys)
        self._client.delete(self.key_registry)
