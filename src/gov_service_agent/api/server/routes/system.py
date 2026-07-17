"""系统和健康检查路由。"""

from __future__ import annotations

from gov_service_agent.api.server.runtime import index_queue, settings, workflow
from gov_service_agent.core.runtime import production_readiness_errors


def register_system_routes(app):
    """注册系统状态路由。"""
    @app.get("/api/declaration-qa/health")
    def health():
        """返回服务健康检查信息。"""
        return {
            "ok": True,
            "service": "gov-service-agent",
            "entrypoint": "gov_service_agent.api.server.app:app",
            "rag_engine": workflow.rag_engine.__class__.__name__,
            "chunks": workflow.rag_engine.chunk_count,
            "cache_backend": workflow.rag_engine.cache_store.backend,
            "vector_backend": workflow.rag_engine.vector_index.backend,
            "keyword_backend": getattr(workflow.rag_engine.keyword_index, "backend", "memory"),
            "external_chunk_store": bool(workflow.rag_engine.chunk_store),
            "business_adapter": workflow.business_adapter.__class__.__name__,
            "audit_store": workflow.audit_store.__class__.__name__,
            "embedding_provider": workflow.rag_engine.embedding_provider.name,
            "rerank_provider": workflow.rag_engine.reranker.name,
            "rerank_model": getattr(workflow.rag_engine.reranker, "model", ""),
            "text_rerank_model": getattr(workflow.rag_engine.reranker, "text_model", ""),
            "query_planning": "agent_tool_schema",
            "agent": workflow.agent.__class__.__name__,
            "agent_text_model": getattr(getattr(workflow.agent, "model", None), "model_name", ""),
            "agent_multimodal_model": getattr(
                getattr(workflow.agent, "multimodal_model", None),
                "model_name",
                "",
            ),
            "answer_generation": "single_agent",
            "agent_max_iterations": workflow.agent_max_iterations,
            "registered_tools": [item["function"]["name"] for item in workflow.tool_registry.definitions()],
            "answer_fallback": "none",
            "max_retrieval_workers": workflow.max_retrieval_workers,
            "auth_required": settings.effective_auth_required,
            "production_ready": not production_readiness_errors(settings),
            "production_readiness_errors": production_readiness_errors(settings),
            "index_queue_backend": index_queue.backend,
        }

    @app.post("/api/declaration-qa/admin/init-schema")
    def init_schema():
        """初始化数据库表结构。"""
        if not hasattr(workflow.audit_store, "init_schema"):
            return {"initialized": False, "message": "PostgreSQL store is not configured"}
        workflow.audit_store.init_schema()
        return {"initialized": True}
