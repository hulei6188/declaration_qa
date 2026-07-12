"""工作流文本安全辅助函数。"""

from __future__ import annotations

import re

def mask_sensitive_text(value: str) -> str:
    """遮盖文本中的敏感证件和手机号信息。"""
    text = str(value or "")
    text = re.sub(r"\b\d{17}[\dXx]\b", "[id_card]", text)
    text = re.sub(r"\b1[3-9]\d{9}\b", "[phone]", text)
    return text


def is_unsafe_query(query: str) -> bool:
    """判断是否为不应回答的高风险问题。"""
    return any(keyword in str(query or "") for keyword in ["伪造", "造假", "假材料", "绕过审核"])
