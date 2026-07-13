"""LangChain model construction compatibility helpers."""

from __future__ import annotations

from typing import Any


def create_chat_openai(**kwargs: Any):
    """Create ChatOpenAI and tolerate malformed process-level proxy variables."""
    import httpx
    from langchain_openai import ChatOpenAI

    try:
        return ChatOpenAI(**kwargs)
    except httpx.InvalidURL:
        base_url = str(kwargs.get("base_url") or "").strip()
        if base_url:
            httpx.URL(base_url)
        return ChatOpenAI(
            **kwargs,
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )
