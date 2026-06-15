from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeclarationQASettings(BaseSettings):
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

    embedding_provider: str = Field(default="hashing", alias="DECLARATION_QA_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-large", alias="DECLARATION_QA_EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=96, alias="DECLARATION_QA_EMBEDDING_DIMENSIONS")

    rerank_provider: str = Field(default="heuristic", alias="DECLARATION_QA_RERANK_PROVIDER")
    rerank_api_url: str = Field(default="", alias="DECLARATION_QA_RERANK_API_URL")
    rerank_api_key: str = Field(default="", alias="DECLARATION_QA_RERANK_API_KEY")
    rerank_model: str = Field(default="", alias="DECLARATION_QA_RERANK_MODEL")

    llm_provider: str = Field(default="deterministic", alias="DECLARATION_QA_LLM_PROVIDER")
    llm_model: str = Field(default="qwen-plus", alias="DECLARATION_QA_LLM_MODEL")
    openai_api_key: str = Field(default="", alias="DECLARATION_QA_OPENAI_API_KEY")
    openai_api_base: str = Field(default="", alias="DECLARATION_QA_OPENAI_API_BASE")
    llm_temperature: float = Field(default=0.1, alias="DECLARATION_QA_LLM_TEMPERATURE")

    business_api_base_url: str = Field(default="", alias="DECLARATION_QA_BUSINESS_API_BASE_URL")
    business_api_token: str = Field(default="", alias="DECLARATION_QA_BUSINESS_API_TOKEN")
    business_api_timeout_seconds: int = Field(default=5, alias="DECLARATION_QA_BUSINESS_API_TIMEOUT_SECONDS")
    business_application_path_template: str = Field(
        default="/applications/{apply_id}",
        alias="DECLARATION_QA_BUSINESS_APPLICATION_PATH_TEMPLATE",
    )
    business_status_path_template: str = Field(
        default="/applications/{apply_id}/status",
        alias="DECLARATION_QA_BUSINESS_STATUS_PATH_TEMPLATE",
    )
    business_reject_path_template: str = Field(
        default="/applications/{apply_id}/reject-info",
        alias="DECLARATION_QA_BUSINESS_REJECT_PATH_TEMPLATE",
    )
    business_access_path_template: str = Field(
        default="/applications/{apply_id}/access",
        alias="DECLARATION_QA_BUSINESS_ACCESS_PATH_TEMPLATE",
    )

    auth_required: bool = Field(default=False, alias="DECLARATION_QA_AUTH_REQUIRED")
    jwt_secret: str = Field(default="", alias="DECLARATION_QA_JWT_SECRET")
    jwt_algorithms: str = Field(default="HS256", alias="DECLARATION_QA_JWT_ALGORITHMS")
    internal_api_key: str = Field(default="", alias="DECLARATION_QA_INTERNAL_API_KEY")

    ocr_api_url: str = Field(default="", alias="DECLARATION_QA_OCR_API_URL")
    ocr_api_key: str = Field(default="", alias="DECLARATION_QA_OCR_API_KEY")

    index_queue_backend: str = Field(default="inline", alias="DECLARATION_QA_INDEX_QUEUE_BACKEND")
    index_queue_name: str = Field(default="declaration-qa-index", alias="DECLARATION_QA_INDEX_QUEUE_NAME")
    index_job_timeout_seconds: int = Field(default=1800, alias="DECLARATION_QA_INDEX_JOB_TIMEOUT_SECONDS")
    upload_dir: str = Field(default="data/declaration_qa/uploads", alias="DECLARATION_QA_UPLOAD_DIR")

    seed_knowledge: bool = Field(default=True, alias="DECLARATION_QA_SEED_KNOWLEDGE")

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in {"prod", "production"}

    @property
    def effective_allow_in_memory_fallbacks(self) -> bool:
        return False if self.is_production else self.allow_in_memory_fallbacks

    @property
    def effective_database_url(self) -> str | None:
        return self.database_url

    @property
    def effective_redis_url(self) -> str | None:
        return self.redis_url

    @property
    def effective_auth_required(self) -> bool:
        return self.auth_required or self.is_production

    @property
    def jwt_algorithm_list(self) -> list[str]:
        return [item.strip() for item in self.jwt_algorithms.split(",") if item.strip()]

    @property
    def effective_upload_dir(self) -> Path:
        return Path(self.upload_dir)


@lru_cache
def get_settings() -> DeclarationQASettings:
    return DeclarationQASettings()
