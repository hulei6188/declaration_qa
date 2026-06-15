from __future__ import annotations

import json
from typing import Any, Protocol


class CacheStore(Protocol):
    backend: str

    def get_json(self, key: str) -> dict[str, Any] | None:
        ...

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        ...

    def clear(self) -> None:
        ...


class InMemoryCacheStore:
    backend = "memory"

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def get_json(self, key: str) -> dict[str, Any] | None:
        item = self._items.get(key)
        return dict(item) if item else None

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        self._items[key] = dict(value)

    def clear(self) -> None:
        self._items.clear()


class RedisCacheStore:
    backend = "redis"
    key_registry = "declaration_qa:cache_keys"

    def __init__(self, redis_url: str, *, allow_fallback: bool = True) -> None:
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
        if not self._available:
            self._fallback.set_json(key, value, ttl_seconds=ttl_seconds)
            return
        self._client.setex(key, max(1, int(ttl_seconds)), json.dumps(value, ensure_ascii=False, default=str))
        self._client.sadd(self.key_registry, key)

    def clear(self) -> None:
        if not self._available:
            self._fallback.clear()
            return
        keys = [str(item) for item in self._client.smembers(self.key_registry)]
        if keys:
            self._client.delete(*keys)
        self._client.delete(self.key_registry)
