from types import SimpleNamespace

from gov_service_agent.indexing.attachments.pipeline import build_vision_summary_client
from gov_service_agent.indexing.document_parsing.vision_summary import HttpVisionSummaryClient


class FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "这是一张取水许可申请表，主要字段包括申请人、取水用途和取水地点。"
                    }
                }
            ]
        }


def test_qwen_vl_image_summary_uses_image_and_ocr_context(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("gov_service_agent.indexing.document_parsing.vision_summary.requests.post", fake_post)
    client = HttpVisionSummaryClient(
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test-key",
        model="qwen3-vl-plus",
        timeout_seconds=45,
        max_tokens=900,
    )

    summary = client.summarize_image(
        filename="取水许可申请书.png",
        content_type="image/png",
        data=b"fake-image",
        ocr_text="申请人：某公司；取水用途：工业用水",
    )

    assert summary.startswith("这是一张取水许可申请表")
    assert captured["url"].endswith("/compatible-mode/v1/chat/completions")
    assert captured["json"]["model"] == "qwen3-vl-plus"
    assert captured["json"]["enable_thinking"] is False
    content = captured["json"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "申请人：某公司" in content[1]["text"]


def test_gpt_5_6_image_summary_uses_openai_chat_url_and_reasoning_none(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("gov_service_agent.indexing.document_parsing.vision_summary.requests.post", fake_post)
    client = HttpVisionSummaryClient(
        url="http://gateway.example/v1",
        api_key="test-key",
        model="gpt-5.6-sol",
    )

    client.summarize_image(
        filename="取水许可申请书.png",
        content_type="image/png",
        data=b"fake-image",
    )

    assert captured["url"] == "http://gateway.example/v1/chat/completions"
    assert captured["json"]["model"] == "gpt-5.6-sol"
    assert captured["json"]["reasoning_effort"] == "none"
    assert "enable_thinking" not in captured["json"]


def test_gpt_5_6_image_summary_reuses_llm_gateway_as_a_credential_pair():
    settings = SimpleNamespace(
        image_summary_enabled=True,
        image_summary_api_url="",
        image_summary_api_key="stale-dashscope-key",
        image_summary_model="gpt-5.6-sol",
        image_summary_timeout_seconds=60,
        image_summary_max_tokens=1200,
        llm_model="gpt-5.6-sol",
        llm_api_base="http://gateway.example/v1",
        llm_api_key="gateway-key",
        ocr_api_url="https://dashscope.aliyuncs.com/api/v1",
        ocr_api_key="ocr-key",
    )

    client = build_vision_summary_client(settings)

    assert client is not None
    assert client.url == "http://gateway.example/v1"
    assert client.api_key == "gateway-key"
