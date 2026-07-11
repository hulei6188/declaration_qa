"""按配置组装申报问答工作流及配套组件。"""

from __future__ import annotations

from declaration_qa.storage.audit import InMemoryAuditStore
from declaration_qa.core.business import InMemoryBusinessDataAdapter
from declaration_qa.storage.cache import InMemoryCacheStore, RedisCacheStore
from declaration_qa.core.config import DeclarationQASettings, get_settings
from declaration_qa.search.keyword import ElasticsearchKeywordIndex
from declaration_qa.memory import ContextMemoryManager, InMemoryContextMemoryStore
from declaration_qa.memory.retrieval import ContextMemoryRetriever
from declaration_qa.rag.model_providers.factories import (
    build_agent_model,
    build_embedding_provider,
    build_reranker,
)
from declaration_qa.storage.repository.store import PostgresDeclarationQAStore
from declaration_qa.rag.engine.external_index import ExternalIndexRAGEngine
from declaration_qa.rag.engine.seed import build_seed_rag_engine
from declaration_qa.core.runtime import assert_production_ready
from declaration_qa.indexing.document_parsing.ocr import HttpOcrClient
from declaration_qa.search.vector.memory import MemoryVectorIndex
from declaration_qa.search.vector.milvus import MilvusVectorIndex
from declaration_qa.application.workflow import DeclarationQAWorkflow


def create_workflow(settings: DeclarationQASettings | None = None) -> DeclarationQAWorkflow:
    """按配置创建完整申报问答工作流。"""
    settings = settings or get_settings()
    assert_production_ready(settings)
    embedding_provider = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    agent_model = build_agent_model(settings)
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
    memory_vector_index = (
        MilvusVectorIndex(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            collection_name=settings.memory_milvus_collection,
            dimension=embedding_provider.dimension,
            allow_fallback=settings.effective_allow_in_memory_fallbacks,
        )
        if settings.context_memory_enabled and settings.vector_backend == "milvus"
        else MemoryVectorIndex()
    )
    keyword_index = (
        ElasticsearchKeywordIndex(
            url=settings.elasticsearch_url,
            api_key=settings.elasticsearch_api_key,
            index_name=settings.elasticsearch_index,
            timeout_seconds=settings.external_request_timeout_seconds,
        )
        if settings.keyword_backend.strip().lower() == "elasticsearch"
        else None
    )
    memory_keyword_index = (
        ElasticsearchKeywordIndex(
            url=settings.elasticsearch_url,
            api_key=settings.elasticsearch_api_key,
            index_name=settings.memory_elasticsearch_index,
            timeout_seconds=settings.external_request_timeout_seconds,
        )
        if settings.context_memory_enabled and settings.keyword_backend.strip().lower() == "elasticsearch"
        else None
    )
    audit_store = (
        PostgresDeclarationQAStore(settings.effective_database_url)
        if settings.effective_database_url
        else InMemoryAuditStore()
    )
    business_adapter = audit_store if isinstance(audit_store, PostgresDeclarationQAStore) else InMemoryBusinessDataAdapter()
    memory_store = audit_store if hasattr(audit_store, "upsert_memory") else InMemoryContextMemoryStore()
    context_memory_manager = ContextMemoryManager(
        cache_store=cache_store,
        memory_store=memory_store,
        retriever=ContextMemoryRetriever(
            store=memory_store,
            vector_index=memory_vector_index,
            keyword_index=memory_keyword_index,
            embedding_provider=embedding_provider,
            distance_threshold=settings.long_memory_distance_threshold,
            keyword_min_score=settings.long_memory_keyword_min_score,
        ),
        short_ttl_seconds=settings.short_memory_ttl_seconds,
        long_ttl_seconds=settings.long_memory_ttl_seconds,
        retrieval_cache_ttl_seconds=settings.long_memory_retrieval_cache_ttl_seconds,
        long_memory_top_k=settings.long_memory_top_k,
        async_workers=settings.long_memory_async_workers,
        enabled=settings.context_memory_enabled,
    )
    use_external_chunk_store = (
        isinstance(audit_store, PostgresDeclarationQAStore)
        and settings.vector_backend.strip().lower() == "milvus"
        and settings.keyword_backend.strip().lower() == "elasticsearch"
    )
    if use_external_chunk_store:
        if keyword_index is None:
            raise RuntimeError("External index RAG requires an Elasticsearch keyword index")
        rag_engine = ExternalIndexRAGEngine(
            cache_store=cache_store,
            vector_index=vector_index,
            keyword_index=keyword_index,
            chunk_store=audit_store,
            embedding_provider=embedding_provider,
            reranker=reranker,
            cache_ttl_seconds=settings.redis_ttl_seconds,
            rerank_score_threshold=settings.rerank_score_threshold,
            rerank_max_sources=settings.rerank_max_sources,
        )
    else:
        rag_engine = build_seed_rag_engine(
            cache_store=cache_store,
            vector_index=vector_index,
            embedding_provider=embedding_provider,
            reranker=reranker,
            keyword_index=keyword_index,
            cache_ttl_seconds=settings.redis_ttl_seconds,
            rerank_score_threshold=settings.rerank_score_threshold,
            rerank_max_sources=settings.rerank_max_sources,
            include_seed=bool(settings.seed_knowledge and not isinstance(audit_store, PostgresDeclarationQAStore)),
        )
    if hasattr(audit_store, "list_indexed_knowledge"):
        if use_external_chunk_store:
            rag_engine.load_indexed_data(audit_store.list_documents(), [])
        else:
            stored_documents, stored_chunks = audit_store.list_indexed_knowledge()
            if stored_chunks:
                rag_engine.load_indexed_data(stored_documents, stored_chunks)
            elif stored_documents and settings.vector_backend != "milvus":
                rag_engine.rebuild(stored_documents)
    elif not settings.seed_knowledge:
        rag_engine.rebuild([])

    query_image_ocr_client = (
        HttpOcrClient(
            url=settings.ocr_api_url,
            api_key=settings.ocr_api_key,
            model=settings.ocr_model,
            timeout_seconds=settings.query_image_ocr_timeout_seconds,
        )
        if settings.query_image_ocr_enabled and settings.ocr_api_url
        else None
    )

    return DeclarationQAWorkflow(
        rag_engine=rag_engine,
        business_adapter=business_adapter,
        audit_store=audit_store,
        agent_model=agent_model,
        agent_max_iterations=settings.agent_max_iterations,
        agent_multimodal_enabled=(
            settings.agent_vl_enabled
            or "vl" in str(settings.agent_model or settings.llm_model).lower()
        ),
        agent_allowed_image_dirs=[
            settings.chat_image_dir,
            settings.attachment_asset_dir,
            settings.upload_dir,
        ],
        agent_max_evidence_images=settings.agent_vl_max_evidence_images,
        agent_min_image_score=settings.agent_vl_min_image_score,
        agent_max_image_bytes=settings.agent_vl_max_image_bytes,
        query_image_ocr_client=query_image_ocr_client,
        query_image_ocr_max_chars=settings.query_image_ocr_max_chars,
        context_memory_manager=context_memory_manager,
        max_retrieval_workers=settings.max_retrieval_workers,
    )


def create_index_job_store(workflow: DeclarationQAWorkflow):
    """根据工作流存储能力创建索引任务存储。"""
    from declaration_qa.indexing.jobs import InMemoryIndexJobStore

    return workflow.audit_store if hasattr(workflow.audit_store, "create_index_job") else InMemoryIndexJobStore()
