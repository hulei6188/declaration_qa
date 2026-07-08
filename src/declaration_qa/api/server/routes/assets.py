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
