"""图片视觉摘要客户端。"""

from __future__ import annotations

import base64
import re

import requests

from gov_service_agent.indexing.document_parsing.models import DocumentParseError
from gov_service_agent.indexing.document_parsing.ocr import (
    dashscope_compatible_chat_url,
    extract_dashscope_message_text,
    response_json_or_error,
)
from gov_service_agent.indexing.document_parsing.text import normalize_extracted_text


class HttpVisionSummaryClient:
    """通过 OpenAI-compatible 视觉模型生成图片检索摘要。"""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 60,
        max_tokens: int = 1200,
    ) -> None:
        """初始化视觉摘要客户端。"""
        self.url = str(url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        self.timeout_seconds = max(1, int(timeout_seconds or 60))
        self.max_tokens = max(128, int(max_tokens or 1200))

    def summarize_image(
        self,
        *,
        filename: str,
        content_type: str,
        data: bytes,
        ocr_text: str = "",
    ) -> str:
        """生成面向政务知识检索的图片摘要。"""
        if not self.url:
            raise DocumentParseError("Image summary requires GOV_SERVICE_AGENT_IMAGE_SUMMARY_API_URL")
        if not self.api_key:
            raise DocumentParseError("Image summary requires GOV_SERVICE_AGENT_IMAGE_SUMMARY_API_KEY")
        if not self.model:
            raise DocumentParseError("Image summary requires GOV_SERVICE_AGENT_IMAGE_SUMMARY_MODEL")
        if not data:
            raise DocumentParseError("Image summary input is empty")

        mime_type = content_type or "image/png"
        image_base64 = base64.b64encode(data).decode("ascii")
        payload = {
            "model": self.model,
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
                            "text": vision_summary_prompt(filename=filename, ocr_text=ocr_text),
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if self.model.lower().startswith("gpt-5.6"):
            payload["reasoning_effort"] = "none"
        else:
            payload["enable_thinking"] = False
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
        summary = clean_vision_summary(extract_dashscope_message_text(response_payload))
        if summary:
            return summary
        raise DocumentParseError("Vision summary response does not contain text")


def vision_summary_prompt(*, filename: str, ocr_text: str = "") -> str:
    """构建适合政务图片入库的视觉摘要提示。"""
    ocr_context = normalize_extracted_text(ocr_text)[:5000]
    prompt = (
        "请分析这张政务申报相关图片，生成一段用于 RAG 检索的中文视觉摘要。"
        f"文件名：{filename or 'document'}。\n"
        "请先判断图片类型，例如流程图、申请表、证件、协议、扫描文档、表格、截图或示意图；"
        "再概括图片主题、主要对象和用途，并保留能够影响检索和回答的关键信息："
        "事项名称、材料名称、字段名称、流程节点、分支条件、办理时限、数量金额、日期、机构、"
        "签字盖章状态及版式关系。流程图要按箭头顺序描述，表格要说明表头和关键行列，"
        "空白表单要列出需要填写的主要字段。"
        "不要臆测被遮挡或无法辨认的内容，不要输出思考过程，不要复述任务说明。"
        "摘要控制在 600 个中文字符以内，使用简洁自然语言，可分行但不要使用 Markdown 代码块。"
    )
    if ocr_context:
        prompt += f"\n以下 OCR 文本只用于辅助核对，请结合图片结构纠错，不要机械照抄：\n{ocr_context}"
    return prompt


def clean_vision_summary(value: str) -> str:
    """清理视觉模型返回的包装文本。"""
    normalized = str(value or "").strip()
    normalized = re.sub(r"^```(?:markdown|md|text)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*```$", "", normalized)
    return normalize_extracted_text(normalized)
