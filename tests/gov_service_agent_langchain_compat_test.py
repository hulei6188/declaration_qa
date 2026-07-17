import httpx

from gov_service_agent.rag.model_providers import langchain_compat


def test_chat_openai_retries_without_environment_proxy(monkeypatch):
    calls = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "http_client" not in kwargs:
                raise httpx.InvalidURL("Invalid port: ':1'")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    result = langchain_compat.create_chat_openai(api_key="test", model="test")

    assert isinstance(result, FakeChatOpenAI)
    assert len(calls) == 2
    assert isinstance(calls[1]["http_client"], httpx.Client)
    assert isinstance(calls[1]["http_async_client"], httpx.AsyncClient)


def test_chat_openai_does_not_hide_an_invalid_model_base_url(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            raise httpx.InvalidURL("invalid URL")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    try:
        langchain_compat.create_chat_openai(api_key="test", model="test", base_url="http://localhost::1")
    except httpx.InvalidURL:
        pass
    else:
        raise AssertionError("invalid model base URL must not be hidden")
