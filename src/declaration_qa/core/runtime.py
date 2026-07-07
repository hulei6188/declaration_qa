"""运行时配置校验和生产环境安全检查。"""

from __future__ import annotations

from declaration_qa.core.config import DeclarationQASettings


def production_readiness_errors(settings: DeclarationQASettings) -> list[str]:
    """返回生产模式下阻止启动的硬性问题。

    开发环境可以使用确定性或本地实现来支持测试和界面开发；
    生产环境必须快速失败，不能静默提供模拟回答或把审计数据写入进程内存。
    """

    if not settings.is_production:
        return []

    errors: list[str] = []
    if not settings.effective_database_url:
        errors.append("DECLARATION_QA_DATABASE_URL is required in production")
    if not settings.effective_redis_url:
        errors.append("DECLARATION_QA_REDIS_URL is required in production")
    if settings.vector_backend.lower() != "milvus":
        errors.append("DECLARATION_QA_VECTOR_BACKEND=milvus is required in production")
    if settings.keyword_backend.lower() != "elasticsearch":
        errors.append("DECLARATION_QA_KEYWORD_BACKEND=elasticsearch is required in production")
    if settings.keyword_backend.lower() == "elasticsearch" and not settings.elasticsearch_url:
        errors.append("DECLARATION_QA_ELASTICSEARCH_URL is required when keyword backend is elasticsearch")
    if not settings.jwt_secret and not settings.internal_api_key:
        errors.append("DECLARATION_QA_JWT_SECRET or DECLARATION_QA_INTERNAL_API_KEY is required in production")
    if settings.llm_provider.lower() == "deterministic":
        errors.append("DECLARATION_QA_LLM_PROVIDER must use a real chat model in production")
    if settings.embedding_provider.lower() == "hashing":
        errors.append("DECLARATION_QA_EMBEDDING_PROVIDER must use a real embedding model in production")
    if settings.rerank_provider.lower() == "heuristic":
        errors.append("DECLARATION_QA_RERANK_PROVIDER must use a real rerank provider in production")
    if settings.index_queue_backend.lower() != "rq":
        errors.append("DECLARATION_QA_INDEX_QUEUE_BACKEND=rq is required in production")
    if settings.seed_knowledge:
        errors.append("DECLARATION_QA_SEED_KNOWLEDGE=false is required in production")
    return errors


def assert_production_ready(settings: DeclarationQASettings) -> None:
    """校验生产环境必需配置是否齐备。"""
    errors = production_readiness_errors(settings)
    if errors:
        raise RuntimeError("Declaration QA production configuration is incomplete: " + "; ".join(errors))
