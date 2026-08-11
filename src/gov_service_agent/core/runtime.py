"""运行时配置校验和生产环境安全检查。"""

from __future__ import annotations

from gov_service_agent.core.config import GovServiceAgentSettings


def production_readiness_errors(settings: GovServiceAgentSettings) -> list[str]:
    """返回生产模式下阻止启动的硬性问题。

    开发环境可以使用确定性或本地实现来支持测试和界面开发；
    生产环境必须快速失败，不能静默提供模拟回答或把审计数据写入进程内存。
    """

    if not settings.is_production:
        return []

    errors: list[str] = []
    if not settings.effective_database_url:
        errors.append("GOV_SERVICE_AGENT_DATABASE_URL is required in production")
    if not settings.effective_redis_url:
        errors.append("GOV_SERVICE_AGENT_REDIS_URL is required in production")
    if settings.vector_backend.lower() != "milvus":
        errors.append("GOV_SERVICE_AGENT_VECTOR_BACKEND=milvus is required in production")
    if settings.keyword_backend.lower() != "elasticsearch":
        errors.append("GOV_SERVICE_AGENT_KEYWORD_BACKEND=elasticsearch is required in production")
    if settings.keyword_backend.lower() == "elasticsearch" and not settings.elasticsearch_url:
        errors.append("GOV_SERVICE_AGENT_ELASTICSEARCH_URL is required when keyword backend is elasticsearch")
    if not settings.jwt_secret and not settings.internal_api_key:
        errors.append("GOV_SERVICE_AGENT_JWT_SECRET or GOV_SERVICE_AGENT_INTERNAL_API_KEY is required in production")
    if settings.llm_provider.lower() == "deterministic":
        errors.append("GOV_SERVICE_AGENT_LLM_PROVIDER must use a real chat model in production")
    if settings.embedding_provider.lower() == "hashing":
        errors.append("GOV_SERVICE_AGENT_EMBEDDING_PROVIDER must use a real embedding model in production")
    if settings.rerank_provider.lower() == "heuristic":
        errors.append("GOV_SERVICE_AGENT_RERANK_PROVIDER must use a real rerank provider in production")
    if settings.index_queue_backend.lower() != "rq":
        errors.append("GOV_SERVICE_AGENT_INDEX_QUEUE_BACKEND=rq is required in production")
    if settings.evaluation_enabled:
        if not settings.effective_evaluation_database_url:
            errors.append("GOV_SERVICE_AGENT_EVAL_DATABASE_URL is required when evaluation is enabled in production")
        elif settings.effective_evaluation_database_url == settings.effective_database_url:
            errors.append("GOV_SERVICE_AGENT_EVAL_DATABASE_URL must differ from GOV_SERVICE_AGENT_DATABASE_URL in production")
        if settings.evaluation_queue_backend.lower() != "rq":
            errors.append("GOV_SERVICE_AGENT_EVAL_QUEUE_BACKEND=rq is required when evaluation is enabled in production")
        if not settings.evaluation_pii_salt:
            errors.append("GOV_SERVICE_AGENT_EVAL_PII_SALT is required when evaluation is enabled in production")
        if settings.evaluation_judge_enabled:
            if not settings.effective_evaluation_judge_model:
                errors.append("GOV_SERVICE_AGENT_EVAL_JUDGE_MODEL is required when evaluation judge is enabled")
            if not settings.effective_evaluation_judge_api_key:
                errors.append("GOV_SERVICE_AGENT_EVAL_JUDGE_API_KEY is required when evaluation judge is enabled")
        if settings.online_evaluation_enabled:
            if not settings.langfuse_enabled or not settings.langfuse_public_key or not settings.langfuse_secret_key:
                errors.append("Langfuse must be configured when online evaluation is enabled")
            if not settings.evaluation_judge_enabled:
                errors.append("GOV_SERVICE_AGENT_EVAL_JUDGE_ENABLED=true is required when online evaluation is enabled")
    if settings.online_evaluation_enabled and not settings.evaluation_enabled:
        errors.append("GOV_SERVICE_AGENT_EVALUATION_ENABLED=true is required when online evaluation is enabled")
    if settings.seed_knowledge:
        errors.append("GOV_SERVICE_AGENT_SEED_KNOWLEDGE=false is required in production")
    return errors


def assert_production_ready(settings: GovServiceAgentSettings) -> None:
    """校验生产环境必需配置是否齐备。"""
    errors = production_readiness_errors(settings)
    if errors:
        raise RuntimeError("Gov Service Agent production configuration is incomplete: " + "; ".join(errors))
