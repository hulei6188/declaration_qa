"""知识库视觉资产访问路由。"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from declaration_qa.api.server.runtime import settings


def register_asset_routes(app):
    """注册附件视觉资产读取接口。"""

    @app.get("/api/declaration-qa/assets/{asset_name}")
    def get_asset(asset_name: str):
        """读取入库时生成的图片资产。"""
        if not re.fullmatch(r"[a-fA-F0-9]{24}\.[A-Za-z0-9]+", asset_name or ""):
            raise HTTPException(status_code=404, detail="asset not found")
        asset_dir = Path(settings.attachment_asset_dir).resolve()
        asset_path = (asset_dir / asset_name).resolve()
        if asset_dir not in asset_path.parents or not asset_path.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(asset_path)

    @app.get("/api/declaration-qa/chat-images/{image_name}")
    def get_chat_image(image_name: str):
        """读取聊天中用户上传的临时图片。"""
        if not re.fullmatch(r"[a-fA-F0-9]{24}\.[A-Za-z0-9]+", image_name or ""):
            raise HTTPException(status_code=404, detail="image not found")
        image_dir = Path(settings.chat_image_dir).resolve()
        image_path = (image_dir / image_name).resolve()
        if image_dir not in image_path.parents or not image_path.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(image_path)
