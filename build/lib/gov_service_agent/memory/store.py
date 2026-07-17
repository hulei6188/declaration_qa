"""长期上下文记忆存储协议和内存实现。"""

from __future__ import annotations

from typing import Any, Protocol

from gov_service_agent.memory.models import ContextMemoryItem


class ContextMemoryStore(Protocol):
    """长期上下文记忆事实存储协议。"""

    def upsert_memory(self, memory: ContextMemoryItem) -> None:
        """写入或更新长期记忆。"""
        ...

    def get_memories_by_ids(self, memory_ids: list[str]) -> list[ContextMemoryItem]:
        """按记忆 ID 批量读取。"""
        ...

    def list_user_memories(self, *, user_id: int, limit: int = 100) -> list[ContextMemoryItem]:
        """读取指定用户最近长期记忆。"""
        ...


class InMemoryContextMemoryStore:
    """进程内长期记忆事实存储。"""

    def __init__(self) -> None:
        """初始化内存存储。"""
        self._items: dict[str, ContextMemoryItem] = {}

    def upsert_memory(self, memory: ContextMemoryItem) -> None:
        """写入或更新长期记忆。"""
        self._items[memory.id] = memory

    def get_memories_by_ids(self, memory_ids: list[str]) -> list[ContextMemoryItem]:
        """按记忆 ID 批量读取。"""
        output = []
        for memory_id in unique_ids(memory_ids):
            item = self._items.get(memory_id)
            if item:
                output.append(item)
        return output

    def list_user_memories(self, *, user_id: int, limit: int = 100) -> list[ContextMemoryItem]:
        """读取指定用户最近长期记忆。"""
        items = [item for item in self._items.values() if int(item.user_id) == int(user_id) and item.status == "published"]
        items = sorted(items, key=lambda item: item.created_at, reverse=True)
        return items[: max(1, int(limit or 100))]


def unique_ids(values: list[Any]) -> list[str]:
    """保序去重 ID 列表。"""
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
