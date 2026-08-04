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
    evaluation_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_EVALUATION_ENABLED")
    evaluation_database_url: str | None = Field(default=None, alias="GOV_SERVICE_AGENT_EVAL_DATABASE_URL")

    redis_url: str | None = Field(default=None, alias="GOV_SERVICE_AGENT_REDIS_URL")
    redis_ttl_seconds: int = Field(default=600, alias="GOV_SERVICE_AGENT_REDIS_TTL_SECONDS")

    memory_summary_max_tokens: int = Field(default=72000, alias="GOV_SERVICE_AGENT_MEMORY_SUMMARY_MAX_TOKENS")
    memory_summary_trigger_tokens: int = Field(default=48000, alias="GOV_SERVICE_AGENT_MEMORY_SUMMARY_TRIGGER_TOKENS")
    memory_summary_output_tokens: int = Field(default=5000, alias="GOV_SERVICE_AGENT_MEMORY_SUMMARY_OUTPUT_TOKENS")
    long_term_memory_top_k: int = Field(default=5, alias="GOV_SERVICE_AGENT_LONG_TERM_MEMORY_TOP_K")
    long_term_memory_workers: int = Field(default=1, alias="GOV_SERVICE_AGENT_LONG_TERM_MEMORY_WORKERS")

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
    text_rerank_model: str = Field(default="qwen3-rerank", alias="GOV_SERVICE_AGENT_TEXT_RERANK_MODEL")
    rerank_score_threshold: float = Field(default=0.0, alias="GOV_SERVICE_AGENT_RERANK_SCORE_THRESHOLD")
    rerank_max_sources: int = Field(default=20, alias="GOV_SERVICE_AGENT_RERANK_MAX_SOURCES")
    retrieval_candidates_per_query: int = Field(default=30, alias="GOV_SERVICE_AGENT_RETRIEVAL_CANDIDATES_PER_QUERY")
    rerank_top_n_per_query: int = Field(default=12, alias="GOV_SERVICE_AGENT_RERANK_TOP_N_PER_QUERY")
    min_sources_per_query: int = Field(default=2, alias="GOV_SERVICE_AGENT_MIN_SOURCES_PER_QUERY")
    max_accumulated_sources: int = Field(default=48, alias="GOV_SERVICE_AGENT_MAX_ACCUMULATED_SOURCES")

    llm_provider: str = Field(default="deterministic", alias="GOV_SERVICE_AGENT_LLM_PROVIDER")
    llm_model: str = Field(default="deepseek-v4-flash", alias="GOV_SERVICE_AGENT_LLM_MODEL")
    llm_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_LLM_API_KEY")
    llm_api_base: str = Field(default="https://api.deepseek.com", alias="GOV_SERVICE_AGENT_LLM_API_BASE")
    openai_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_OPENAI_API_KEY")
    openai_api_base: str = Field(default="", alias="GOV_SERVICE_AGENT_OPENAI_API_BASE")
    external_request_timeout_seconds: float = Field(default=10.0, alias="GOV_SERVICE_AGENT_EXTERNAL_REQUEST_TIMEOUT_SECONDS")
    agent_model: str = Field(default="", alias="GOV_SERVICE_AGENT_AGENT_MODEL")
    agent_timeout_seconds: float = Field(default=60.0, alias="GOV_SERVICE_AGENT_AGENT_TIMEOUT_SECONDS")
    agent_max_iterations: int = Field(default=4, alias="GOV_SERVICE_AGENT_AGENT_MAX_ITERATIONS")
    agent_vl_enabled: bool = Field(default=False, alias="GOV_SERVICE_AGENT_AGENT_VL_ENABLED")
    agent_vl_max_evidence_images: int = Field(default=3, alias="GOV_SERVICE_AGENT_AGENT_VL_MAX_EVIDENCE_IMAGES")
    agent_vl_min_image_score: float = Field(default=0.7, alias="GOV_SERVICE_AGENT_AGENT_VL_MIN_IMAGE_SCORE")
    agent_vl_max_image_bytes: int = Field(default=10 * 1024 * 1024, alias="GOV_SERVICE_AGENT_AGENT_VL_MAX_IMAGE_BYTES")
    max_retrieval_workers: int = Field(default=4, alias="GOV_SERVICE_AGENT_MAX_RETRIEVAL_WORKERS")
    langfuse_enabled: bool = Field(default=False, alias="GOV_SERVICE_AGENT_LANGFUSE_ENABLED")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(default="http://localhost:3000", alias="LANGFUSE_BASE_URL")

    ragas_enabled: bool = Field(default=False, alias="GOV_SERVICE_AGENT_RAGAS_ENABLED")
    ragas_timeout_seconds: int = Field(default=60, alias="GOV_SERVICE_AGENT_RAGAS_TIMEOUT_SECONDS")
    ragas_max_retries: int = Field(default=1, alias="GOV_SERVICE_AGENT_RAGAS_MAX_RETRIES")
    evaluation_queue_backend: str = Field(default="inline", alias="GOV_SERVICE_AGENT_EVAL_QUEUE_BACKEND")
    evaluation_queue_name: str = Field(default="gov-service-agent-evaluation", alias="GOV_SERVICE_AGENT_EVAL_QUEUE_NAME")
    evaluation_job_timeout_seconds: int = Field(default=14400, alias="GOV_SERVICE_AGENT_EVAL_JOB_TIMEOUT_SECONDS")
    evaluation_judge_enabled: bool = Field(default=False, alias="GOV_SERVICE_AGENT_EVAL_JUDGE_ENABLED")
    evaluation_judge_model: str = Field(default="", alias="GOV_SERVICE_AGENT_EVAL_JUDGE_MODEL")
    evaluation_judge_api_base: str = Field(default="", alias="GOV_SERVICE_AGENT_EVAL_JUDGE_API_BASE")
    evaluation_judge_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_EVAL_JUDGE_API_KEY")
    evaluation_pii_salt: str = Field(default="", alias="GOV_SERVICE_AGENT_EVAL_PII_SALT")
    evaluation_candidate_scan_limit: int = Field(default=500, alias="GOV_SERVICE_AGENT_EVAL_CANDIDATE_SCAN_LIMIT")
    online_evaluation_enabled: bool = Field(default=False, alias="GOV_SERVICE_AGENT_ONLINE_EVAL_ENABLED")
    online_evaluation_interval_minutes: int = Field(default=15, ge=1, le=1440, alias="GOV_SERVICE_AGENT_ONLINE_EVAL_INTERVAL_MINUTES")
    online_evaluation_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0, alias="GOV_SERVICE_AGENT_ONLINE_EVAL_SAMPLE_RATE")
    online_evaluation_sample_limit: int = Field(default=50, ge=1, le=2000, alias="GOV_SERVICE_AGENT_ONLINE_EVAL_SAMPLE_LIMIT")
    online_evaluation_lookback_hours: int = Field(default=24, ge=1, le=8760, alias="GOV_SERVICE_AGENT_ONLINE_EVAL_LOOKBACK_HOURS")
    online_evaluation_faithfulness_threshold: float = Field(default=0.9, ge=0.0, le=1.0, alias="GOV_SERVICE_AGENT_ONLINE_EVAL_FAITHFULNESS_THRESHOLD")
    online_evaluation_run_on_start: bool = Field(default=True, alias="GOV_SERVICE_AGENT_ONLINE_EVAL_RUN_ON_START")

    auth_required: bool = Field(default=False, alias="GOV_SERVICE_AGENT_AUTH_REQUIRED")
    jwt_secret: str = Field(default="", alias="GOV_SERVICE_AGENT_JWT_SECRET")
    jwt_algorithms: str = Field(default="HS256", alias="GOV_SERVICE_AGENT_JWT_ALGORITHMS")
    internal_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_INTERNAL_API_KEY")

    ocr_api_url: str = Field(default="", alias="GOV_SERVICE_AGENT_OCR_API_URL")
    ocr_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_OCR_API_KEY")
    ocr_model: str = Field(default="", alias="GOV_SERVICE_AGENT_OCR_MODEL")
    query_image_ocr_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_QUERY_IMAGE_OCR_ENABLED")
    query_image_ocr_max_chars: int = Field(default=4000, alias="GOV_SERVICE_AGENT_QUERY_IMAGE_OCR_MAX_CHARS")
    query_image_ocr_timeout_seconds: int = Field(default=30, alias="GOV_SERVICE_AGENT_QUERY_IMAGE_OCR_TIMEOUT_SECONDS")
    image_summary_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_IMAGE_SUMMARY_ENABLED")
    image_summary_api_url: str = Field(default="", alias="GOV_SERVICE_AGENT_IMAGE_SUMMARY_API_URL")
    image_summary_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_IMAGE_SUMMARY_API_KEY")
    image_summary_model: str = Field(default="qwen3-vl-plus", alias="GOV_SERVICE_AGENT_IMAGE_SUMMARY_MODEL")
    image_summary_max_tokens: int = Field(default=1200, alias="GOV_SERVICE_AGENT_IMAGE_SUMMARY_MAX_TOKENS")
    image_summary_timeout_seconds: int = Field(default=60, alias="GOV_SERVICE_AGENT_IMAGE_SUMMARY_TIMEOUT_SECONDS")

    document_parser_provider: str = Field(default="auto", alias="GOV_SERVICE_AGENT_DOCUMENT_PARSER_PROVIDER")
    document_parser_order: str = Field(default="mineru,native", alias="GOV_SERVICE_AGENT_DOCUMENT_PARSER_ORDER")
    document_parser_fallback_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_DOCUMENT_PARSER_FALLBACK_ENABLED")
    legacy_doc_conversion_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_LEGACY_DOC_CONVERSION_ENABLED")
    legacy_doc_converter_command: str = Field(default="", alias="GOV_SERVICE_AGENT_LEGACY_DOC_CONVERTER_COMMAND")
    legacy_doc_conversion_timeout_seconds: int = Field(default=120, alias="GOV_SERVICE_AGENT_LEGACY_DOC_CONVERSION_TIMEOUT_SECONDS")
    mineru_api_url: str = Field(default="http://127.0.0.1:8011", alias="GOV_SERVICE_AGENT_MINERU_API_URL")
    mineru_backend: str = Field(default="pipeline", alias="GOV_SERVICE_AGENT_MINERU_BACKEND")
    mineru_method: str = Field(default="auto", alias="GOV_SERVICE_AGENT_MINERU_METHOD")
    mineru_language: str = Field(default="ch", alias="GOV_SERVICE_AGENT_MINERU_LANGUAGE")
    mineru_model_source: str = Field(default="modelscope", alias="GOV_SERVICE_AGENT_MINERU_MODEL_SOURCE")
    mineru_formula_enabled: bool = Field(default=False, alias="GOV_SERVICE_AGENT_MINERU_FORMULA_ENABLED")
    mineru_timeout_seconds: int = Field(default=3600, alias="GOV_SERVICE_AGENT_MINERU_TIMEOUT_SECONDS")
    chonkie_text_max_chars: int = Field(default=900, alias="GOV_SERVICE_AGENT_CHONKIE_TEXT_MAX_CHARS")
    chonkie_text_min_chars: int = Field(default=120, alias="GOV_SERVICE_AGENT_CHONKIE_TEXT_MIN_CHARS")
    chonkie_table_max_chars: int = Field(default=3500, alias="GOV_SERVICE_AGENT_CHONKIE_TABLE_MAX_CHARS")
    llamaparse_api_key: str = Field(default="", alias="GOV_SERVICE_AGENT_LLAMAPARSE_API_KEY")
    llamaparse_base_url: str = Field(default="", alias="GOV_SERVICE_AGENT_LLAMAPARSE_BASE_URL")
    llamaparse_result_type: str = Field(default="markdown", alias="GOV_SERVICE_AGENT_LLAMAPARSE_RESULT_TYPE")

    attachment_indexing_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_ATTACHMENT_INDEXING_ENABLED")
    attachment_cache_dir: str = Field(default="data/declaration_qa/attachments", alias="GOV_SERVICE_AGENT_ATTACHMENT_CACHE_DIR")
    attachment_asset_dir: str = Field(default="data/declaration_qa/assets", alias="GOV_SERVICE_AGENT_ATTACHMENT_ASSET_DIR")
    attachment_download_timeout_seconds: float = Field(default=15.0, alias="GOV_SERVICE_AGENT_ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS")
    attachment_download_retries: int = Field(default=1, alias="GOV_SERVICE_AGENT_ATTACHMENT_DOWNLOAD_RETRIES")
    attachment_max_bytes: int = Field(default=20 * 1024 * 1024, alias="GOV_SERVICE_AGENT_ATTACHMENT_MAX_BYTES")
    attachment_max_per_document: int = Field(default=30, alias="GOV_SERVICE_AGENT_ATTACHMENT_MAX_PER_DOCUMENT")
    attachment_max_workers: int = Field(default=3, alias="GOV_SERVICE_AGENT_ATTACHMENT_MAX_WORKERS")
    attachment_pdf_ocr_fallback_enabled: bool = Field(default=True, alias="GOV_SERVICE_AGENT_ATTACHMENT_PDF_OCR_FALLBACK_ENABLED")
    attachment_pdf_ocr_max_pages: int = Field(default=8, alias="GOV_SERVICE_AGENT_ATTACHMENT_PDF_OCR_MAX_PAGES")
    attachment_pdf_ocr_dpi: int = Field(default=180, alias="GOV_SERVICE_AGENT_ATTACHMENT_PDF_OCR_DPI")

    index_queue_backend: str = Field(default="inline", alias="GOV_SERVICE_AGENT_INDEX_QUEUE_BACKEND")
    index_queue_name: str = Field(default="declaration-qa-index", alias="GOV_SERVICE_AGENT_INDEX_QUEUE_NAME")
    index_job_timeout_seconds: int = Field(default=7200, alias="GOV_SERVICE_AGENT_INDEX_JOB_TIMEOUT_SECONDS")
    upload_dir: str = Field(default="data/declaration_qa/uploads", alias="GOV_SERVICE_AGENT_UPLOAD_DIR")
    chat_image_dir: str = Field(default="data/declaration_qa/query_images", alias="GOV_SERVICE_AGENT_CHAT_IMAGE_DIR")
    chat_image_max_bytes: int = Field(default=8 * 1024 * 1024, alias="GOV_SERVICE_AGENT_CHAT_IMAGE_MAX_BYTES")
    application_material_dir: str = Field(
        default="data/declaration_qa/application_materials",
        alias="GOV_SERVICE_AGENT_APPLICATION_MATERIAL_DIR",
    )
    application_material_max_bytes: int = Field(
        default=20 * 1024 * 1024,
        alias="GOV_SERVICE_AGENT_APPLICATION_MATERIAL_MAX_BYTES",
    )

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
    def effective_evaluation_database_url(self) -> str | None:
        """返回评测工作流专用数据库连接地址。"""
        return self.evaluation_database_url

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
