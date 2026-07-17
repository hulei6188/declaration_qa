"""政务服务智能体服务的环境配置定义。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GovServiceAgentSettings(BaseSettings):
    """政务服务智能体服务运行所需的环境变量配置。"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="development", alias="GOV_SERVICE_AGENT_ENV")
    allow_in_memory_fallbacks: bool = Field(default=True, alias="GOV_SERVICE_AGENT_ALLOW_IN_MEMORY_FALLBACKS")

    database_url: str | None = Field(default=None, alias="GOV_SERVICE_AGENT_DATABASE_URL")

    redis_url: str | None = Field(default=None, alias="GOV_SERVICE_AGENT_REDIS_URL")
    redis_ttl_seconds: int = Field(default=600, alias="GOV_SERVICE_AGENT_REDIS_TTL_SECONDS")

    context_memory_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_CONTEXT_MEMORY_ENABLED")
    short_memory_ttl_seconds: int = Field(default=3600, alias="GOV_SERVICE_AGENT_SHORT_MEMORY_TTL_SECONDS")
    long_memory_ttl_seconds: int = Field(default=604800, alias="GOV_SERVICE_AGENT_LONG_MEMORY_TTL_SECONDS")
    long_memory_retrieval_cache_ttl_seconds: int = Field(default=300, alias="GOV_SERVICE_AGENT_LONG_MEMORY_RETRIEVAL_CACHE_TTL_SECONDS")
    long_memory_async_workers: int = Field(default=2, alias="GOV_SERVICE_AGENT_LONG_MEMORY_ASYNC_WORKERS")
    long_memory_top_k: int = Field(default=5, alias="GOV_SERVICE_AGENT_LONG_MEMORY_TOP_K")
    long_memory_distance_threshold: float = Field(default=0.4, alias="GOV_SERVICE_AGENT_LONG_MEMORY_DISTANCE_THRESHOLD")
    long_memory_keyword_min_score: float = Field(default=0.2, alias="GOV_SERVICE_AGENT_LONG_MEMORY_KEYWORD_MIN_SCORE")
    memory_milvus_collection: str = Field(default="declaration_qa_context_memories", alias="GOV_SERVICE_AGENT_MEMORY_MILVUS_COLLECTION")
    memory_elasticsearch_index: str = Field(default="declaration_qa_context_memories", alias="GOV_SERVICE_AGENT_MEMORY_ELASTICSEARCH_INDEX")

    vector_backend: str = Field(default="memory", alias="GOV_SERVICE_AGENT_VECTOR_BACKEND")
    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="GOV_SERVICE_AGENT_MILVUS_URI")
    milvus_token: str = Field(default="", alias="GOV_SERVICE_AGENT_MILVUS_TOKEN")
    milvus_collection: str = Field(default="declaration_qa_chunks", alias="GOV_SERVICE_AGENT_MILVUS_COLLECTION")

    keyword_backend: str = Field(default="memory", alias="GOV_SERVICE_AGENT_KEYWORD_BACKEND")
    elasticsearch_url: str = Field(default="", alias="GOV_SERVICE_AGENT_ELASTICSEARCH_URL")
    elasticsearch_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_ELASTICSEARCH_API_KEY")
    elasticsearch_index: str = Field(default="declaration_qa_chunks", alias="GOV_SERVICE_AGENT_ELASTICSEARCH_INDEX")

    embedding_provider: str = Field(default="hashing", alias="GOV_SERVICE_AGENT_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-large", alias="GOV_SERVICE_AGENT_EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=96, alias="GOV_SERVICE_AGENT_EMBEDDING_DIMENSIONS")
    embedding_image_res_level: int = Field(default=0, alias="GOV_SERVICE_AGENT_EMBEDDING_IMAGE_RES_LEVEL")

    rerank_provider: str = Field(default="heuristic", alias="GOV_SERVICE_AGENT_RERANK_PROVIDER")
    rerank_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_RERANK_API_KEY")
    rerank_model: str = Field(default="qwen3-vl-rerank", alias="GOV_SERVICE_AGENT_RERANK_MODEL")
    rerank_score_threshold: float = Field(default=0.0, alias="GOV_SERVICE_AGENT_RERANK_SCORE_THRESHOLD")
    rerank_max_sources: int = Field(default=12, alias="GOV_SERVICE_AGENT_RERANK_MAX_SOURCES")

    llm_provider: str = Field(default="deterministic", alias="GOV_SERVICE_AGENT_LLM_PROVIDER")
    llm_model: str = Field(default="qwen-plus", alias="GOV_SERVICE_AGENT_LLM_MODEL")
    openai_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_OPENAI_API_KEY")
    openai_api_base: str = Field(default="", alias="GOV_SERVICE_AGENT_OPENAI_API_BASE")
    llm_temperature: float = Field(default=0.1, alias="GOV_SERVICE_AGENT_LLM_TEMPERATURE")
    external_request_timeout_seconds: float = Field(default=10.0, alias="GOV_SERVICE_AGENT_EXTERNAL_REQUEST_TIMEOUT_SECONDS")
    question_resolver_model: str = Field(default="", alias="GOV_SERVICE_AGENT_QUESTION_RESOLVER_MODEL")
    query_rewriter_model: str = Field(default="", alias="GOV_SERVICE_AGENT_QUERY_REWRITER_MODEL")
    max_sub_question_workers: int = Field(default=4, alias="GOV_SERVICE_AGENT_MAX_SUB_QUESTION_WORKERS")
    max_retrieval_workers: int = Field(default=4, alias="GOV_SERVICE_AGENT_MAX_RETRIEVAL_WORKERS")
    langfuse_enabled: bool = Field(default=False, alias="GOV_SERVICE_AGENT_LANGFUSE_ENABLED")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(default="http://localhost:3000", alias="LANGFUSE_BASE_URL")

    auth_required: bool = Field(default=False, alias="GOV_SERVICE_AGENT_AUTH_REQUIRED")
    jwt_secret: str = Field(default="", alias="GOV_SERVICE_AGENT_JWT_SECRET")
    jwt_algorithms: str = Field(default="HS256", alias="GOV_SERVICE_AGENT_JWT_ALGORITHMS")
    internal_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_INTERNAL_API_KEY")

    ocr_api_url: str = Field(default="", alias="GOV_SERVICE_AGENT_OCR_API_URL")
    ocr_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_OCR_API_KEY")
    ocr_model: str = Field(default="", alias="GOV_SERVICE_AGENT_OCR_MODEL")

    attachment_indexing_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_ATTACHMENT_INDEXING_ENABLED")
    attachment_cache_dir: str = Field(default="data/declaration_qa/attachments", alias="GOV_SERVICE_AGENT_ATTACHMENT_CACHE_DIR")
    attachment_asset_dir: str = Field(default="data/declaration_qa/assets", alias="GOV_SERVICE_AGENT_ATTACHMENT_ASSET_DIR")
    attachment_download_timeout_seconds: float = Field(default=15.0, alias="GOV_SERVICE_AGENT_ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS")
    attachment_download_retries: int = Field(default=1, alias="GOV_SERVICE_AGENT_ATTACHMENT_DOWNLOAD_RETRIES")
    attachment_max_bytes: int = Field(default=20 * 1024 * 1024, alias="GOV_SERVICE_AGENT_ATTACHMENT_MAX_BYTES")
    attachment_max_per_document: int = Field(default=30, alias="GOV_SERVICE_AGENT_ATTACHMENT_MAX_PER_DOCUMENT")
    attachment_pdf_ocr_fallback_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_ATTACHMENT_PDF_OCR_FALLBACK_ENABLED")
    attachment_pdf_ocr_max_pages: int = Field(default=8, alias="GOV_SERVICE_AGENT_ATTACHMENT_PDF_OCR_MAX_PAGES")
    attachment_pdf_ocr_dpi: int = Field(default=180, alias="GOV_SERVICE_AGENT_ATTACHMENT_PDF_OCR_DPI")

    index_queue_backend: str = Field(default="inline", alias="GOV_SERVICE_AGENT_INDEX_QUEUE_BACKEND")
    index_queue_name: str = Field(default="declaration-qa-index", alias="GOV_SERVICE_AGENT_INDEX_QUEUE_NAME")
    index_job_timeout_seconds: int = Field(default=1800, alias="GOV_SERVICE_AGENT_INDEX_JOB_TIMEOUT_SECONDS")
    upload_dir: str = Field(default="data/declaration_qa/uploads", alias="GOV_SERVICE_AGENT_UPLOAD_DIR")
    chat_image_dir: str = Field(default="data/declaration_qa/query_images", alias="GOV_SERVICE_AGENT_CHAT_IMAGE_DIR")
    chat_image_max_bytes: int = Field(default=8 * 1024 * 1024, alias="GOV_SERVICE_AGENT_CHAT_IMAGE_MAX_BYTES")

    seed_knowledge: bool = Field(default=True, alias="GOV_SERVICE_AGENT_SEED_KNOWLEDGE")

    @property
    def is_production(self) -> bool:
        """判断当前运行环境是否为生产环境。"""
        return self.environment.strip().lower() in {"prod", "production"}

    @property
    def effective_allow_in_memory_fallbacks(self) -> bool:
        """计算当前环境是否允许内存兜底。"""
        return False if self.is_production else self.allow_in_memory_fallbacks

    @property
    def effective_database_url(self) -> str | None:
        """返回当前生效的数据库连接地址。"""
        return self.database_url

    @property
    def effective_redis_url(self) -> str | None:
        """返回当前生效的 Redis 连接地址。"""
        return self.redis_url

    @property
    def effective_auth_required(self) -> bool:
        """计算当前环境是否强制鉴权。"""
        return self.auth_required or self.is_production

    @property
    def jwt_algorithm_list(self) -> list[str]:
        """解析 JWT 算法配置为列表。"""
        return [item.strip() for item in self.jwt_algorithms.split(",") if item.strip()]

    @property
    def effective_upload_dir(self) -> Path:
        """返回上传文件存储目录。"""
        return Path(self.upload_dir)


@lru_cache
def get_settings() -> GovServiceAgentSettings:
    """创建并缓存服务配置对象。"""
    return GovServiceAgentSettings()
