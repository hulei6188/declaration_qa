"""长期上下文记忆格式化工具。"""

from __future__ import annotations

import json
from typing import Any


def format_long_term_context(state: dict[str, Any]) -> str:
    """格式化长期上下文记忆，供提示词使用。"""
    context = state.get("long_term_context") if isinstance(state.get("long_term_context"), dict) else {}
    if not context:
        context = {"items": [], "status": {"retrieval_mode": "none", "reason": "not_loaded"}}
    return json.dumps(context, ensure_ascii=False, indent=2)
