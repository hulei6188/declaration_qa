-- Isolated schema for the Declaration QA Workflow service.
--
-- This file is intentionally separate from AgentBase's main schema. The
-- declaration QA service starts from declaration_qa.server:app and stores its
-- own knowledge, chat, audit, and feedback data under the declaration_qa schema.

BEGIN;

CREATE SCHEMA IF NOT EXISTS declaration_qa;

CREATE TABLE IF NOT EXISTS declaration_qa.knowledge_bases (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(160) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_by BIGINT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_knowledge_bases_created_by
    ON declaration_qa.knowledge_bases (created_by);

CREATE TABLE IF NOT EXISTS declaration_qa.documents (
    id BIGSERIAL PRIMARY KEY,
    document_uid VARCHAR(120) NOT NULL DEFAULT '',
    knowledge_base_id BIGINT NOT NULL REFERENCES declaration_qa.knowledge_bases(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL DEFAULT '',
    content_type VARCHAR(120) NOT NULL DEFAULT 'text/plain',
    source_type VARCHAR(40) NOT NULL CHECK (
        source_type IN ('policy', 'guide', 'faq', 'process', 'material_rule', 'operation_notice')
    ),
    source_url TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL,
    text_preview TEXT NOT NULL DEFAULT '',
    content_hash VARCHAR(80) NOT NULL DEFAULT '',
    biz_type VARCHAR(80) NOT NULL,
    region_code VARCHAR(20) NOT NULL,
    policy_version VARCHAR(80) NOT NULL,
    material_code VARCHAR(80) NOT NULL DEFAULT '',
    reject_code VARCHAR(80) NOT NULL DEFAULT '',
    audit_node VARCHAR(80) NOT NULL DEFAULT '',
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'indexing', 'published', 'failed', 'deprecated')),
    version_group_uid VARCHAR(120) NOT NULL DEFAULT '',
    version_no INTEGER NOT NULL DEFAULT 1 CHECK (version_no > 0),
    rollout_percent INTEGER NOT NULL DEFAULT 100 CHECK (rollout_percent >= 0 AND rollout_percent <= 100),
    change_note TEXT NOT NULL DEFAULT '',
    published_at TIMESTAMPTZ NULL,
    deprecated_at TIMESTAMPTZ NULL,
    effective_start_time DATE NULL,
    effective_end_time DATE NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    error_message TEXT NOT NULL DEFAULT '',
    created_by BIGINT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (effective_end_time IS NULL OR effective_start_time IS NULL OR effective_end_time > effective_start_time)
);
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS document_uid VARCHAR(120) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS version_group_uid VARCHAR(120) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS version_no INTEGER NOT NULL DEFAULT 1;
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS rollout_percent INTEGER NOT NULL DEFAULT 100;
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS change_note TEXT NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ NULL;
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS deprecated_at TIMESTAMPTZ NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_documents_uid
    ON declaration_qa.documents (document_uid)
    WHERE document_uid <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_kb
    ON declaration_qa.documents (knowledge_base_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_version_group
    ON declaration_qa.documents (version_group_uid, version_no DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_policy_filter
    ON declaration_qa.documents (biz_type, region_code, policy_version, status);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_effective_time
    ON declaration_qa.documents (effective_start_time, effective_end_time);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_reject_filter
    ON declaration_qa.documents (biz_type, region_code, material_code, reject_code, audit_node);

CREATE TABLE IF NOT EXISTS declaration_qa.chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES declaration_qa.documents(id) ON DELETE CASCADE,
    parent_chunk_id BIGINT NULL REFERENCES declaration_qa.chunks(id) ON DELETE CASCADE,
    chunk_uid VARCHAR(180) NOT NULL UNIQUE,
    chunk_type VARCHAR(20) NOT NULL CHECK (chunk_type IN ('parent', 'child')),
    chunk_index INTEGER NOT NULL DEFAULT 0 CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    content_hash VARCHAR(80) NOT NULL DEFAULT '',
    token_count INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
    embedding_id VARCHAR(180) NOT NULL DEFAULT '',
    vector_collection VARCHAR(160) NOT NULL DEFAULT '',
    biz_type VARCHAR(80) NOT NULL,
    region_code VARCHAR(20) NOT NULL,
    policy_version VARCHAR(80) NOT NULL,
    source_type VARCHAR(40) NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    material_code VARCHAR(80) NOT NULL DEFAULT '',
    reject_code VARCHAR(80) NOT NULL DEFAULT '',
    audit_node VARCHAR(80) NOT NULL DEFAULT '',
    status VARCHAR(20) NOT NULL DEFAULT 'published' CHECK (status IN ('draft', 'published', 'deprecated')),
    effective_start_time DATE NULL,
    effective_end_time DATE NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (effective_end_time IS NULL OR effective_start_time IS NULL OR effective_end_time > effective_start_time)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_document
    ON declaration_qa.chunks (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_parent
    ON declaration_qa.chunks (parent_chunk_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_policy_filter
    ON declaration_qa.chunks (biz_type, region_code, policy_version, status);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_effective_time
    ON declaration_qa.chunks (effective_start_time, effective_end_time);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_reject_filter
    ON declaration_qa.chunks (biz_type, region_code, material_code, reject_code, audit_node);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_source_type
    ON declaration_qa.chunks (source_type);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_metadata_gin
    ON declaration_qa.chunks USING GIN (metadata);

UPDATE declaration_qa.documents
SET source_type = 'material_rule'
WHERE source_type = 'reject_rule';

UPDATE declaration_qa.chunks
SET
    source_type = 'material_rule',
    metadata = jsonb_set(metadata, '{source_type}', '"material_rule"', true)
WHERE source_type = 'reject_rule';

DO $$
DECLARE
    old_source_type_constraint TEXT;
BEGIN
    SELECT conname
    INTO old_source_type_constraint
    FROM pg_constraint
    WHERE conrelid = 'declaration_qa.documents'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) LIKE '%source_type%'
      AND pg_get_constraintdef(oid) LIKE '%reject_rule%'
    LIMIT 1;

    IF old_source_type_constraint IS NOT NULL THEN
        EXECUTE format('ALTER TABLE declaration_qa.documents DROP CONSTRAINT %I', old_source_type_constraint);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'declaration_qa.documents'::regclass
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%source_type%'
          AND pg_get_constraintdef(oid) NOT LIKE '%reject_rule%'
    ) THEN
        ALTER TABLE declaration_qa.documents
            ADD CONSTRAINT ck_declaration_qa_documents_source_type
            CHECK (source_type IN ('policy', 'guide', 'faq', 'process', 'material_rule', 'operation_notice'));
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS declaration_qa.index_jobs (
    id BIGSERIAL PRIMARY KEY,
    job_uid VARCHAR(80) NOT NULL UNIQUE,
    knowledge_base_id BIGINT NULL REFERENCES declaration_qa.knowledge_bases(id) ON DELETE SET NULL,
    document_id BIGINT NULL REFERENCES declaration_qa.documents(id) ON DELETE SET NULL,
    job_type VARCHAR(40) NOT NULL CHECK (job_type IN ('document_index', 'knowledge_base_reindex', 'document_delete')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')),
    progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    message TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_index_jobs_status
    ON declaration_qa.index_jobs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_index_jobs_document
    ON declaration_qa.index_jobs (document_id);

CREATE TABLE IF NOT EXISTS declaration_qa.sessions (
    id BIGSERIAL PRIMARY KEY,
    session_uid VARCHAR(80) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    region_code VARCHAR(20) NOT NULL,
    biz_type VARCHAR(80) NULL,
    apply_id VARCHAR(80) NOT NULL DEFAULT '',
    title VARCHAR(200) NOT NULL DEFAULT '申报智能问答',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_sessions_user
    ON declaration_qa.sessions (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_sessions_apply
    ON declaration_qa.sessions (apply_id);

CREATE TABLE IF NOT EXISTS declaration_qa.messages (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES declaration_qa.sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    intent VARCHAR(40) NOT NULL DEFAULT '',
    refused BOOLEAN NOT NULL DEFAULT FALSE,
    refuse_reason VARCHAR(120) NOT NULL DEFAULT '',
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    suggested_actions JSONB NOT NULL DEFAULT '[]'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_session
    ON declaration_qa.messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_intent
    ON declaration_qa.messages (intent);

CREATE TABLE IF NOT EXISTS declaration_qa.audit_logs (
    id BIGSERIAL PRIMARY KEY,
    audit_uid VARCHAR(80) NOT NULL UNIQUE,
    session_id BIGINT NULL REFERENCES declaration_qa.sessions(id) ON DELETE SET NULL,
    user_id BIGINT NOT NULL,
    query TEXT NOT NULL,
    normalized_query TEXT NOT NULL DEFAULT '',
    intent VARCHAR(40) NOT NULL,
    region_code VARCHAR(20) NOT NULL,
    biz_type VARCHAR(80) NULL,
    apply_id VARCHAR(80) NOT NULL DEFAULT '',
    business_status JSONB NOT NULL DEFAULT '{}'::jsonb,
    reject_info JSONB NOT NULL DEFAULT '{}'::jsonb,
    material_mapping JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_filter JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieved_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    rag_status JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence_ok BOOLEAN NOT NULL DEFAULT FALSE,
    refused BOOLEAN NOT NULL DEFAULT FALSE,
    refuse_reason VARCHAR(120) NOT NULL DEFAULT '',
    final_prompt TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_user
    ON declaration_qa.audit_logs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_intent
    ON declaration_qa.audit_logs (intent, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_sources_gin
    ON declaration_qa.audit_logs USING GIN (retrieved_sources);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_filter_gin
    ON declaration_qa.audit_logs USING GIN (retrieval_filter);

CREATE TABLE IF NOT EXISTS declaration_qa.feedback (
    id BIGSERIAL PRIMARY KEY,
    audit_log_id BIGINT NOT NULL REFERENCES declaration_qa.audit_logs(id) ON DELETE CASCADE,
    user_id BIGINT NULL,
    rating VARCHAR(20) NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_audit
    ON declaration_qa.feedback (audit_log_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_rating
    ON declaration_qa.feedback (rating, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.eval_datasets (
    id BIGSERIAL PRIMARY KEY,
    dataset_uid VARCHAR(80) NOT NULL UNIQUE,
    name VARCHAR(160) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_by BIGINT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_datasets_created
    ON declaration_qa.eval_datasets (created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.eval_cases (
    id BIGSERIAL PRIMARY KEY,
    case_uid VARCHAR(80) NOT NULL UNIQUE,
    dataset_id BIGINT NOT NULL REFERENCES declaration_qa.eval_datasets(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    user_id BIGINT NOT NULL DEFAULT 1,
    region_code VARCHAR(20) NOT NULL DEFAULT '340000',
    policy_version VARCHAR(80) NOT NULL DEFAULT '2025',
    biz_type VARCHAR(80) NULL,
    apply_id VARCHAR(80) NULL,
    apply_context JSONB NULL,
    expected_intent VARCHAR(40) NOT NULL DEFAULT '',
    expected_refused BOOLEAN NULL,
    expected_doc_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    expected_answer_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_cases_dataset
    ON declaration_qa.eval_cases (dataset_id, created_at);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_cases_tags
    ON declaration_qa.eval_cases USING GIN (tags);

CREATE TABLE IF NOT EXISTS declaration_qa.eval_runs (
    id BIGSERIAL PRIMARY KEY,
    run_uid VARCHAR(80) NOT NULL UNIQUE,
    dataset_id BIGINT NOT NULL REFERENCES declaration_qa.eval_datasets(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'succeeded', 'failed')),
    note TEXT NOT NULL DEFAULT '',
    top_k INTEGER NOT NULL DEFAULT 4 CHECK (top_k > 0),
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ NULL
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_runs_dataset
    ON declaration_qa.eval_runs (dataset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_runs_status
    ON declaration_qa.eval_runs (status, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.eval_run_items (
    id BIGSERIAL PRIMARY KEY,
    item_uid VARCHAR(80) NOT NULL UNIQUE,
    run_id BIGINT NOT NULL REFERENCES declaration_qa.eval_runs(id) ON DELETE CASCADE,
    case_id BIGINT NULL REFERENCES declaration_qa.eval_cases(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    expected_intent VARCHAR(40) NOT NULL DEFAULT '',
    actual_intent VARCHAR(40) NOT NULL DEFAULT '',
    expected_refused BOOLEAN NULL,
    actual_refused BOOLEAN NOT NULL DEFAULT FALSE,
    expected_doc_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    retrieved_doc_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    expected_answer_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer TEXT NOT NULL DEFAULT '',
    audit_uid VARCHAR(80) NOT NULL DEFAULT '',
    recall_hit BOOLEAN NULL,
    refusal_match BOOLEAN NULL,
    intent_match BOOLEAN NULL,
    keyword_hit BOOLEAN NULL,
    annotation_label VARCHAR(40) NOT NULL DEFAULT '',
    annotation_comment TEXT NOT NULL DEFAULT '',
    annotated_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_run_items_run
    ON declaration_qa.eval_run_items (run_id, id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_run_items_audit
    ON declaration_qa.eval_run_items (audit_uid);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_run_items_annotation
    ON declaration_qa.eval_run_items (annotation_label, created_at DESC);

COMMIT;
