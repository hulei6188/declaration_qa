"""按配置组装申报问答工作流及配套组件。"""

from __future__ import annotations

from declaration_qa.audit import InMemoryAuditStore
from declaration_qa.business import InMemoryBusinessDataAdapter
from declaration_qa.cache import InMemoryCacheStore, RedisCacheStore
from declaration_qa.config import DeclarationQASettings, get_settings
from declaration_qa.model_providers import build_answer_generator, build_embedding_provider, build_question_planner, build_reranker
from declaration_qa.repository import PostgresDeclarationQAStore
from declaration_qa.rag_engine import build_seed_rag_engine
from declaration_qa.runtime import assert_production_ready
from declaration_qa.vector_index import MemoryVectorIndex, MilvusVectorIndex
from declaration_qa.workflow import DeclarationQAWorkflow


def create_workflow(settings: DeclarationQASettings | None = None) -> DeclarationQAWorkflow:
    """按配置创建完整申报问答工作流。"""
    settings = settings or get_settings()
    assert_production_ready(settings)
    embedding_provider = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    answer_generator = build_answer_generator(settings)
    question_planner = build_question_planner(settings)
    cache_store = (
        RedisCacheStore(
            settings.effective_redis_url,
            allow_fallback=settings.effective_allow_in_memory_fallbacks,
        )
        if settings.effective_redis_url
        else InMemoryCacheStore()
    )
    vector_index = (
        MilvusVectorIndex(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            collection_name=settings.milvus_collection,
            dimension=embedding_provider.dimension,
            allow_fallback=settings.effective_allow_in_memory_fallbacks,
        )
        if settings.vector_backend == "milvus"
        else MemoryVectorIndex()
    )
    audit_store = (
        PostgresDeclarationQAStore(settings.effective_database_url)
        if settings.effective_database_url
        else InMemoryAuditStore()
    )
    business_adapter = audit_store if isinstance(audit_store, PostgresDeclarationQAStore) else InMemoryBusinessDataAdapter()
    rag_engine = build_seed_rag_engine(
        cache_store=cache_store,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
        reranker=reranker,
        cache_ttl_seconds=settings.redis_ttl_seconds,
        include_seed=bool(settings.seed_knowledge and not isinstance(audit_store, PostgresDeclarationQAStore)),
    )
    if hasattr(audit_store, "list_indexed_knowledge"):
        stored_documents, stored_chunks = audit_store.list_indexed_knowledge()
        if stored_chunks:
            rag_engine.load_indexed_data(stored_documents, stored_chunks)
        elif stored_documents and settings.vector_backend != "milvus":
            rag_engine.rebuild(stored_documents)
    elif not settings.seed_knowledge:
        rag_engine.rebuild([])

    return DeclarationQAWorkflow(
        rag_engine=rag_engine,
        business_adapter=business_adapter,
        audit_store=audit_store,
        answer_generator=answer_generator,
        question_planner=question_planner,
    )


def create_index_job_store(workflow: DeclarationQAWorkflow):
    """根据工作流存储能力创建索引任务存储。"""
    from declaration_qa.indexing import InMemoryIndexJobStore

    return workflow.audit_store if hasattr(workflow.audit_store, "create_index_job") else InMemoryIndexJobStore()
