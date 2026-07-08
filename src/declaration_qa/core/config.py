"""申报问答服务的环境配置定义。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeclarationQASettings(BaseSettings):
    """申报问答服务运行所需的环境变量配置。"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="development", alias="DECLARATION_QA_ENV")
    allow_in_memory_fallbacks: bool = Field(default=True, alias="DECLARATION_QA_ALLOW_IN_MEMORY_FALLBACKS")

    database_url: str | None = Field(default=None, alias="DECLARATION_QA_DATABASE_URL")

    redis_url: str | None = Field(default=None, alias="DECLARATION_QA_REDIS_URL")
    redis_ttl_seconds: int = Field(default=600, alias="DECLARATION_QA_REDIS_TTL_SECONDS")

    vector_backend: str = Field(default="memory", alias="DECLARATION_QA_VECTOR_BACKEND")
    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="DECLARATION_QA_MILVUS_URI")
    milvus_token: str = Field(default="", alias="DECLARATION_QA_MILVUS_TOKEN")
    milvus_collection: str = Field(default="declaration_qa_chunks", alias="DECLARATION_QA_MILVUS_COLLECTION")

    keyword_backend: str = Field(default="memory", alias="DECLARATION_QA_KEYWORD_BACKEND")
    elasticsearch_url: str = Field(default="", alias="DECLARATION_QA_ELASTICSEARCH_URL")
    elasticsearch_api_key: str = Field(default="", alias="DECLARATION_QA_ELASTICSEARCH_API_KEY")
    elasticsearch_index: str = Field(default="declaration_qa_chunks", alias="DECLARATION_QA_ELASTICSEARCH_INDEX")

    embedding_provider: str = Field(default="hashing", alias="DECLARATION_QA_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-large", alias="DECLARATION_QA_EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=96, alias="DECLARATION_QA_EMBEDDING_DIMENSIONS")

    rerank_provider: str = Field(default="heuristic", alias="DECLARATION_QA_RERANK_PROVIDER")
    rerank_api_key: str = Field(default="", alias="DECLARATION_QA_RERANK_API_KEY")
    rerank_model: str = Field(default="qwen3-rerank", alias="DECLARATION_QA_RERANK_MODEL")
    rerank_score_threshold: float = Field(default=0.0, alias="DECLARATION_QA_RERANK_SCORE_THRESHOLD")
    rerank_max_sources: int = Field(default=12, alias="DECLARATION_QA_RERANK_MAX_SOURCES")

    llm_provider: str = Field(default="deterministic", alias="DECLARATION_QA_LLM_PROVIDER")
    llm_model: str = Field(default="qwen-plus", alias="DECLARATION_QA_LLM_MODEL")
    openai_api_key: str = Field(default="", alias="DECLARATION_QA_OPENAI_API_KEY")
    openai_api_base: str = Field(default="", alias="DECLARATION_QA_OPENAI_API_BASE")
    llm_temperature: float = Field(default=0.1, alias="DECLARATION_QA_LLM_TEMPERATURE")
    external_request_timeout_seconds: float = Field(default=10.0, alias="DECLARATION_QA_EXTERNAL_REQUEST_TIMEOUT_SECONDS")
    question_resolver_model: str = Field(default="", alias="DECLARATION_QA_QUESTION_RESOLVER_MODEL")
    query_rewriter_model: str = Field(default="", alias="DECLARATION_QA_QUERY_REWRITER_MODEL")
    max_sub_question_workers: int = Field(default=4, alias="DECLARATION_QA_MAX_SUB_QUESTION_WORKERS")
    max_retrieval_workers: int = Field(default=4, alias="DECLARATION_QA_MAX_RETRIEVAL_WORKERS")
    langfuse_enabled: bool = Field(default=False, alias="DECLARATION_QA_LANGFUSE_ENABLED")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(default="http://localhost:3000", alias="LANGFUSE_BASE_URL")

    auth_required: bool = Field(default=False, alias="DECLARATION_QA_AUTH_REQUIRED")
    jwt_secret: str = Field(default="", alias="DECLARATION_QA_JWT_SECRET")
    jwt_algorithms: str = Field(default="HS256", alias="DECLARATION_QA_JWT_ALGORITHMS")
    internal_api_key: str = Field(default="", alias="DECLARATION_QA_INTERNAL_API_KEY")

    ocr_api_url: str = Field(default="", alias="DECLARATION_QA_OCR_API_URL")
    ocr_api_key: str = Field(default="", alias="DECLARATION_QA_OCR_API_KEY")
    ocr_model: str = Field(default="", alias="DECLARATION_QA_OCR_MODEL")

    attachment_indexing_enabled: bool = Field(default=True, alias="DECLARATION_QA_ATTACHMENT_INDEXING_ENABLED")
    attachment_cache_dir: str = Field(default="data/declaration_qa/attachments", alias="DECLARATION_QA_ATTACHMENT_CACHE_DIR")
    attachment_download_timeout_seconds: float = Field(default=15.0, alias="DECLARATION_QA_ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS")
    attachment_download_retries: int = Field(default=1, alias="DECLARATION_QA_ATTACHMENT_DOWNLOAD_RETRIES")
    attachment_max_bytes: int = Field(default=20 * 1024 * 1024, alias="DECLARATION_QA_ATTACHMENT_MAX_BYTES")
    attachment_max_per_document: int = Field(default=30, alias="DECLARATION_QA_ATTACHMENT_MAX_PER_DOCUMENT")
    attachment_pdf_ocr_fallback_enabled: bool = Field(default=True, alias="DECLARATION_QA_ATTACHMENT_PDF_OCR_FALLBACK_ENABLED")
    attachment_pdf_ocr_max_pages: int = Field(default=8, alias="DECLARATION_QA_ATTACHMENT_PDF_OCR_MAX_PAGES")
    attachment_pdf_ocr_dpi: int = Field(default=180, alias="DECLARATION_QA_ATTACHMENT_PDF_OCR_DPI")

    index_queue_backend: str = Field(default="inline", alias="DECLARATION_QA_INDEX_QUEUE_BACKEND")
    index_queue_name: str = Field(default="declaration-qa-index", alias="DECLARATION_QA_INDEX_QUEUE_NAME")
    index_job_timeout_seconds: int = Field(default=1800, alias="DECLARATION_QA_INDEX_JOB_TIMEOUT_SECONDS")
    upload_dir: str = Field(default="data/declaration_qa/uploads", alias="DECLARATION_QA_UPLOAD_DIR")

    seed_knowledge: bool = Field(default=True, alias="DECLARATION_QA_SEED_KNOWLEDGE")

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
def get_settings() -> DeclarationQASettings:
    """创建并缓存服务配置对象。"""
    return DeclarationQASettings()
