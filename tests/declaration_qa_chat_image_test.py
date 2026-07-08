import base64
from pathlib import Path

from declaration_qa.api.schemas import ChatImagePayload, ChatRequest
from declaration_qa.api.server.chat_images import prepare_chat_image_request
from declaration_qa.core.config import DeclarationQASettings


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
