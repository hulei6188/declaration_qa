"""申报问答共享路由常量和轻量规则工具。"""

from __future__ import annotations

import json
import re
from typing import Any


ROUTE_KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"

def parse_json_object(value: str) -> dict[str, Any]:
    """从模型输出中解析 JSON 对象。"""
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object")
    return parsed
