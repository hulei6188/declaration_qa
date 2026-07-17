"""按配置组装政务服务智能体工作流及配套组件。"""

from __future__ import annotations

from gov_service_agent.storage.audit import InMemoryAuditStore
from gov_service_agent.core.business import InMemoryBusinessDataAdapter
from gov_service_agent.storage.cache import InMemoryCacheStore, RedisCacheStore
from gov_service_agent.core.config import GovServiceAgentSettings, get_settings
from gov_service_agent.search.keyword import ElasticsearchKeywordIndex
from gov_service_agent.memory import LangGraphMemoryRuntime, build_memory_resources
from gov_service_agent.rag.model_providers.factories import (
    build_agent_model,
    build_embedding_provider,
    build_reranker,
)
from gov_service_agent.storage.repository.store import PostgresGovServiceAgentStore
from gov_service_agent.rag.engine.external_index import ExternalIndexRAGEngine
from gov_service_agent.rag.engine.seed import build_seed_rag_engine
from gov_service_agent.core.runtime import assert_production_ready
from gov_service_agent.indexing.document_parsing.ocr import HttpOcrClient
from gov_service_agent.search.vector.memory import MemoryVectorIndex
from gov_service_agent.search.vector.milvus import MilvusVectorIndex
from gov_service_agent.application.workflow import GovServiceAgentWorkflow


def create_workflow(
    settings: GovServiceAgentSettings | None = None,
    *,
    runtime_database_url: str | None = None,
    knowledge_database_url: str | None = None,
) -> GovServiceAgentWorkflow:
    """按配置创建完整政务服务智能体工作流。"""
    settings = settings or get_settings()
    assert_production_ready(settings)
    runtime_database_url = runtime_database_url if runtime_database_url is not None else settings.effective_database_url
    knowledge_database_url = knowledge_database_url if knowledge_database_url is not None else settings.effective_database_url
    embedding_provider = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    text_agent_model = build_agent_model(
        settings,
        timeout_seconds=settings.agent_timeout_seconds,
        model_name=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
    )
    multimodal_model_name = settings.agent_model or settings.llm_model
    multimodal_agent_model = (
        text_agent_model
        if multimodal_model_name == settings.llm_model
        else build_agent_model(
            settings,
            timeout_seconds=settings.agent_timeout_seconds,
            model_name=multimodal_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
    )
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
    audit_store = (
        PostgresGovServiceAgentStore(runtime_database_url)
        if runtime_database_url
        else InMemoryAuditStore()
    )
    knowledge_store = (
        audit_store
        if not knowledge_database_url or knowledge_database_url == runtime_database_url
        else PostgresGovServiceAgentStore(knowledge_database_url)
    )
    business_adapter = audit_store if isinstance(audit_store, PostgresGovServiceAgentStore) else InMemoryBusinessDataAdapter()
    use_external_chunk_store = (
        isinstance(knowledge_store, PostgresGovServiceAgentStore)
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
            chunk_store=knowledge_store,
            embedding_provider=embedding_provider,
            reranker=reranker,
            cache_ttl_seconds=settings.redis_ttl_seconds,
            rerank_score_threshold=settings.rerank_score_threshold,
            rerank_max_sources=settings.rerank_max_sources,
            retrieval_candidates_per_query=settings.retrieval_candidates_per_query,
            rerank_top_n_per_query=settings.rerank_top_n_per_query,
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
            retrieval_candidates_per_query=settings.retrieval_candidates_per_query,
            rerank_top_n_per_query=settings.rerank_top_n_per_query,
            include_seed=bool(settings.seed_knowledge and not isinstance(audit_store, PostgresGovServiceAgentStore)),
        )
    if hasattr(knowledge_store, "list_indexed_knowledge"):
        if use_external_chunk_store:
            rag_engine.load_indexed_data(knowledge_store.list_documents(), [])
        else:
            stored_documents, stored_chunks = knowledge_store.list_indexed_knowledge()
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

    memory_runtime = LangGraphMemoryRuntime(
        resources=build_memory_resources(
            database_url=runtime_database_url,
            embedding_provider=embedding_provider,
        ),
        model=text_agent_model,
        summary_max_tokens=settings.memory_summary_max_tokens,
        summary_trigger_tokens=settings.memory_summary_trigger_tokens,
        summary_output_tokens=settings.memory_summary_output_tokens,
        long_term_top_k=settings.long_term_memory_top_k,
        background_workers=settings.long_term_memory_workers,
    )
    try:
        created_workflow = GovServiceAgentWorkflow(
            rag_engine=rag_engine,
            business_adapter=business_adapter,
            audit_store=audit_store,
            agent_model=text_agent_model,
            agent_multimodal_model=multimodal_agent_model,
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
            memory_runtime=memory_runtime,
            max_retrieval_workers=settings.max_retrieval_workers,
            min_sources_per_query=settings.min_sources_per_query,
            max_accumulated_sources=settings.max_accumulated_sources,
        )
        created_workflow.managed_stores = [
            store for store in (audit_store, knowledge_store) if isinstance(store, PostgresGovServiceAgentStore)
        ]
        return created_workflow
    except Exception:
        memory_runtime.close()
        raise


def create_evaluation_workflow(settings: GovServiceAgentSettings | None = None) -> GovServiceAgentWorkflow:
    """创建写入独立评测库、只读复用生产知识库的工作流。"""
    settings = settings or get_settings()
    if not settings.effective_evaluation_database_url:
        raise ValueError("GOV_SERVICE_AGENT_EVAL_DATABASE_URL is required")
    if settings.is_production and settings.effective_evaluation_database_url == settings.effective_database_url:
        raise ValueError("evaluation database must differ from production database")
    return create_workflow(
        settings,
        runtime_database_url=settings.effective_evaluation_database_url,
        knowledge_database_url=settings.effective_database_url,
    )


def create_index_job_store(workflow: GovServiceAgentWorkflow):
    """根据工作流存储能力创建索引任务存储。"""
    from gov_service_agent.indexing.jobs import InMemoryIndexJobStore

    return workflow.audit_store if hasattr(workflow.audit_store, "create_index_job") else InMemoryIndexJobStore()
