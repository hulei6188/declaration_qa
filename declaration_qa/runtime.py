from __future__ import annotations

from declaration_qa.config import DeclarationQASettings


def production_readiness_errors(settings: DeclarationQASettings) -> list[str]:
    """Return hard startup blockers for production mode.

    Development can use deterministic/local implementations for tests and local
    UI work. Production must fail fast instead of silently serving mock answers
    or writing audit data to process memory.
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
    if not settings.jwt_secret and not settings.internal_api_key:
        errors.append("DECLARATION_QA_JWT_SECRET or DECLARATION_QA_INTERNAL_API_KEY is required in production")
    if settings.llm_provider.lower() == "deterministic":
        errors.append("DECLARATION_QA_LLM_PROVIDER must use a real chat model in production")
    if settings.intent_router_provider.lower() in {"rule", "rules", "deterministic"}:
        errors.append("DECLARATION_QA_INTENT_ROUTER_PROVIDER must use llm or auto in production")
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
    errors = production_readiness_errors(settings)
    if errors:
        raise RuntimeError("Declaration QA production configuration is incomplete: " + "; ".join(errors))
