"""在线申报材料上传接口。"""

from __future__ import annotations

from fastapi import Depends, File, Form, HTTPException, UploadFile, status

from gov_service_agent.api.auth import RequestContext
from gov_service_agent.api.schemas import ApplicationMaterialUploadResponse
from gov_service_agent.api.server.dependencies import auth_context
from gov_service_agent.api.server.runtime import settings, workflow
from gov_service_agent.application.application_materials import ApplicationMaterialStore


material_store = ApplicationMaterialStore(
    settings.application_material_dir,
    max_bytes=settings.application_material_max_bytes,
)


def register_application_material_routes(app):
    """注册申报材料上传端点。"""

    @app.post(
        "/api/declaration-qa/application-materials/upload",
        response_model=ApplicationMaterialUploadResponse,
    )
    async def upload_application_material(
        file: UploadFile = File(...),
        session_id: str = Form(...),
        user_id: int = Form(default=1),
        item_code: str = Form(default=""),
        field_name: str = Form(...),
        material_key: str = Form(...),
        context: RequestContext = Depends(auth_context),
    ):
        effective_user_id = int(context.user_id or user_id)
        if settings.effective_auth_required and not context.user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user_id is required")
        task = workflow._checkpoint_state(user_id=effective_user_id, session_id=session_id).get("dialog_task") or {}
        schema = task.get("schema") if isinstance(task.get("schema"), dict) else {}
        material_field = next(
            (
                field
                for field in schema.get("fields") or []
                if isinstance(field, dict)
                and str(field.get("name") or "") == str(field_name or "")
                and str(field.get("type") or "") == "material_group"
            ),
            {},
        )
        materials = material_field.get("materials") if isinstance(material_field.get("materials"), list) else []
        valid_material_keys = {
            str(material.get("key") or "")
            for material in materials
            if isinstance(material, dict)
        }
        if (
            task.get("status") != "collecting"
            or str(task.get("item_code") or "") != str(item_code or "")
            or not material_field
            or str(material_key or "") not in valid_material_keys
        ):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="当前表单不接受该申报材料")
        data = await file.read()
        try:
            return material_store.save(
                data=data,
                filename=file.filename or "material.bin",
                content_type=file.content_type or "application/octet-stream",
                user_id=effective_user_id,
                session_id=session_id,
                item_code=item_code,
                field_name=field_name,
                material_key=material_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
