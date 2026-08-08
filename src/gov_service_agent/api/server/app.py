"""应用创建入口。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gov_service_agent import __version__
from gov_service_agent.api.server.routes.audit import register_audit_routes
from gov_service_agent.api.server.routes.assets import register_asset_routes
from gov_service_agent.api.server.routes.application_materials import register_application_material_routes
from gov_service_agent.api.server.routes.chat import register_chat_routes
from gov_service_agent.api.server.routes.evaluation import register_evaluation_routes
from gov_service_agent.api.server.routes.knowledge import register_knowledge_routes
from gov_service_agent.api.server.routes.human_service import register_human_service_routes
from gov_service_agent.api.server.routes.sessions import register_session_routes
from gov_service_agent.api.server.routes.system import register_system_routes


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    app = FastAPI(
        title="Gov Service Agent",
        version=__version__,
        description="Agentic RAG service for government service guidance and workflows.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5174",
            "http://localhost:5174",
            "http://127.0.0.1:8020",
            "http://localhost:8020",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_system_routes(app)
    register_asset_routes(app)
    register_application_material_routes(app)
    register_chat_routes(app)
    register_knowledge_routes(app)
    register_session_routes(app)
    register_audit_routes(app)
    register_evaluation_routes(app)
    register_human_service_routes(app)
    return app


app = create_app()
