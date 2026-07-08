import base64
from pathlib import Path

from declaration_qa.api.schemas import ChatImagePayload, ChatRequest
from declaration_qa.api.server.chat_images import prepare_chat_image_request
from declaration_qa.core.config import DeclarationQASettings
from declaration_qa.storage.audit import InMemoryAuditStore


def test_prepare_chat_image_request_persists_data_url(tmp_path):
    data = b"fake-png"
    request = ChatRequest(
        query="这张图片是什么材料",
        query_image=ChatImagePayload(
            filename="material.png",
            content_type="image/png",
            data_url=f"data:image/png;base64,{base64.b64encode(data).decode()}",
            size_bytes=len(data),
            text_present=False,
        ),
    )
    settings = DeclarationQASettings(
        _env_file=None,
        DECLARATION_QA_CHAT_IMAGE_DIR=str(tmp_path),
        DECLARATION_QA_CHAT_IMAGE_MAX_BYTES=1024,
    )

    prepared = prepare_chat_image_request(request, settings)

    assert prepared.query_image is not None
    assert prepared.query_image.data_url is None
    assert prepared.query_image.content_type == "image/png"
    assert prepared.query_image.size_bytes == len(data)
    assert prepared.query_image.text_present is False
    assert prepared.query_image.file_path
    assert Path(prepared.query_image.file_path).read_bytes() == data


def test_session_history_restores_chat_image_attachment(tmp_path):
    image_path = tmp_path / "aabbccddeeff001122334455.png"
    image_path.write_bytes(b"fake-png")
    store = InMemoryAuditStore()
    store.append(
        {
            "user_id": 1,
            "session_id": "s1",
            "query": "",
            "query_image": {
                "filename": "流程图.png",
                "content_type": "image/png",
                "file_path": str(image_path),
                "size_bytes": 8,
                "text_present": False,
            },
            "answer": "已找到流程图。",
        }
    )

    session = store.get_session(user_id=1, session_id="s1")

    assert session is not None
    user_message = session["messages"][0]
    assert user_message["content"] == ""
    assert user_message["attachments"][0]["filename"] == "流程图.png"
    assert user_message["attachments"][0]["url"] == "/api/declaration-qa/chat-images/aabbccddeeff001122334455.png"
