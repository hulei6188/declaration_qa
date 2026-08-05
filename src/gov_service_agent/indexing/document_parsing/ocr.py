"""HTTP OCR 客户端。"""

from __future__ import annotations

import base64
import re

import requests

from gov_service_agent.indexing.document_parsing.models import DocumentParseError
from gov_service_agent.indexing.document_parsing.text import normalize_extracted_text


class HttpOcrClient:
    """通过外部 HTTP OCR 服务提取图片文本。"""

    def __init__(self, *, url: str, api_key: str = "", model: str = "", timeout_seconds: int = 30) -> None:
        """初始化 OCR 客户端。"""
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def extract_text(self, *, filename: str, content_type: str, data: bytes) -> str:
        """调用 OCR 服务从文件字节中提取文本。"""
        if not self.url:
            raise DocumentParseError("Image OCR requires GOV_SERVICE_AGENT_OCR_API_URL")
        if self.uses_dashscope_qwen_ocr():
            return self.extract_text_with_dashscope_qwen_ocr(filename=filename, content_type=content_type, data=data)
        return self.extract_text_with_multipart(filename=filename, content_type=content_type, data=data)

    def extract_text_with_multipart(self, *, filename: str, content_type: str, data: bytes) -> str:
        """调用普通 multipart OCR 服务。"""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        form_data = {"model": self.model} if self.model else None
        response = requests.post(
            self.url,
            headers=headers,
            data=form_data,
            files={"file": (filename, data, content_type or "application/octet-stream")},
            timeout=self.timeout_seconds,
        )
        payload = response_json_or_error(response)
        data_payload = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        text = data_payload.get("text") or data_payload.get("content") or data_payload.get("result")
        if text:
            return str(text)
        raise DocumentParseError("OCR response does not contain extracted text")

    def uses_dashscope_qwen_ocr(self) -> bool:
        """判断当前配置是否为 DashScope Qwen-OCR。"""
        url = self.url.strip().lower()
        model = self.model.strip().lower()
        if not model.startswith("qwen"):
            return False
        return any(token in url for token in ["dashscope.aliyuncs.com", "maas.aliyuncs.com", "compatible-mode"])

    def extract_text_with_dashscope_qwen_ocr(self, *, filename: str, content_type: str, data: bytes) -> str:
        """通过 DashScope OpenAI-compatible 视觉接口执行 OCR。"""
        mime_type = content_type or "image/png"
        image_base64 = base64.b64encode(data).decode("ascii")
        payload = {
            "model": self.model or "qwen-vl-ocr-latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                        },
                        {
                            "type": "text",
                            "text": dashscope_ocr_prompt(filename),
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }
        response = requests.post(
            dashscope_compatible_chat_url(self.url),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response_payload = response_json_or_error(response)
        text = clean_dashscope_ocr_text(extract_dashscope_message_text(response_payload))
        if text:
            return text
        raise DocumentParseError("DashScope OCR response does not contain extracted text")


def response_json_or_error(response: requests.Response) -> dict:
    """读取 HTTP JSON 响应，失败时保留可诊断错误。"""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = str(getattr(response, "text", "") or "")[:500]
        raise DocumentParseError(f"OCR HTTP {response.status_code}: {body}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        body = str(getattr(response, "text", "") or "")[:500]
        raise DocumentParseError(f"OCR response is not valid JSON: {body}") from exc
    return payload if isinstance(payload, dict) else {"data": payload}


def dashscope_compatible_chat_url(url: str) -> str:
    """将 DashScope 或通用 OpenAI-compatible 基础地址转换为聊天接口地址。"""
    base = str(url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/compatible-mode/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/api/v1"):
        return f"{base[:-len('/api/v1')]}/compatible-mode/v1/chat/completions"
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/compatible-mode/v1/chat/completions"


def dashscope_ocr_prompt(filename: str) -> str:
    """返回宽松的 OCR 提示，避免格式约束导致模型返回空内容。"""
    return "请提取图片中的全部文字。"


def extract_dashscope_message_text(payload: dict) -> str:
    """从 DashScope/OpenAI-compatible 响应中提取文本。"""
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(text.strip() for text in texts if text.strip()).strip()
    return ""


def clean_dashscope_ocr_text(text: str) -> str:
    """清理 DashScope OCR 偶尔返回的代码块或 HTML 包装。"""
    normalized = str(text or "").strip()
    normalized = re.sub(r"^```(?:html|xml|markdown|md|text)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*```$", "", normalized)
    if re.search(r"</?(?:html|body|p|div|h[1-6]|table|tr|td|th|li|br)\b", normalized, flags=re.IGNORECASE):
        normalized = re.sub(r"<br\s*/?>", "\n", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"</(?:p|div|h[1-6]|tr|li|table)>", "\n", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"</(?:td|th)>", " | ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"<[^>]+>", " ", normalized)
    return normalize_extracted_text(normalized)
