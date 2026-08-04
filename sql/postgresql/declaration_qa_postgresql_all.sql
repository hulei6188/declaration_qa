-- 合并后的 PostgreSQL SQL 文件
-- 来源目录：C:\Users\Ivy\declaration-qa\sql\postgresql
-- 生成时间：2026-07-13 12:25:23 +08:00
-- 文件顺序：按文件名升序排列

-- ============================================================================
-- 来源：003_declaration_qa_schema.sql
-- ============================================================================

-- 申报问答工作流服务使用的独立数据库模式。
--
-- 此文件有意与 AgentBase 的主数据库模式分离。申报问答服务从
-- gov_service_agent.api.server.app:app 启动，并在 declaration_qa 模式下存储
-- 自身的知识、聊天、审计和反馈数据。

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

CREATE TABLE IF NOT EXISTS declaration_qa.business_applications (
    id BIGSERIAL PRIMARY KEY,
    apply_uid VARCHAR(120) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    item_code VARCHAR(160) NOT NULL DEFAULT '',
    item_name VARCHAR(255) NOT NULL DEFAULT '',
    biz_type VARCHAR(80) NOT NULL,
    region_code VARCHAR(20) NOT NULL DEFAULT '340000',
    status VARCHAR(40) NOT NULL DEFAULT '',
    current_node VARCHAR(80) NOT NULL DEFAULT '',
    node_name VARCHAR(120) NOT NULL DEFAULT '',
    submitted_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    deleted_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_applications_user
    ON declaration_qa.business_applications (user_id, apply_uid)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_applications_filter
    ON declaration_qa.business_applications (biz_type, region_code, status)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_applications_user_item
    ON declaration_qa.business_applications (user_id, item_code, updated_at DESC)
    WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS declaration_qa.business_application_statuses (
    id BIGSERIAL PRIMARY KEY,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    sequence_no INTEGER NOT NULL DEFAULT 1,
    from_status VARCHAR(40) NULL,
    status VARCHAR(40) NOT NULL,
    current_node VARCHAR(80) NOT NULL DEFAULT '',
    node_name VARCHAR(120) NOT NULL DEFAULT '',
    reason_code VARCHAR(80) NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    material_code VARCHAR(80) NOT NULL DEFAULT '',
    material_name VARCHAR(160) NOT NULL DEFAULT '',
    source_event_id VARCHAR(160) NOT NULL DEFAULT '',
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_business_status_sequence
    ON declaration_qa.business_application_statuses (apply_id, sequence_no);
DROP INDEX IF EXISTS declaration_qa.ux_declaration_qa_business_status_source_event;
CREATE UNIQUE INDEX ux_declaration_qa_business_status_source_event
    ON declaration_qa.business_application_statuses (apply_id, source_event_id)
    WHERE source_event_id <> '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_business_status_current
    ON declaration_qa.business_application_statuses (apply_id)
    WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_status_apply
    ON declaration_qa.business_application_statuses (apply_id, created_at DESC);
CREATE OR REPLACE FUNCTION declaration_qa.prepare_business_application_status_event()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    previous_sequence INTEGER;
    previous_status VARCHAR(40);
BEGIN
    PERFORM pg_advisory_xact_lock(NEW.apply_id);

    IF NEW.source_event_id <> ''
       AND EXISTS (
           SELECT 1
           FROM declaration_qa.business_application_statuses
           WHERE apply_id = NEW.apply_id
             AND source_event_id = NEW.source_event_id
       ) THEN
        RETURN NULL;
    END IF;

    SELECT sequence_no, status
    INTO previous_sequence, previous_status
    FROM declaration_qa.business_application_statuses
    WHERE apply_id = NEW.apply_id
    ORDER BY sequence_no DESC, occurred_at DESC, id DESC
    LIMIT 1;

    IF NEW.sequence_no IS NULL
       OR NEW.sequence_no <= COALESCE(previous_sequence, 0) THEN
        NEW.sequence_no := COALESCE(previous_sequence, 0) + 1;
    END IF;
    NEW.from_status := COALESCE(NEW.from_status, previous_status);
    NEW.occurred_at := COALESCE(NEW.occurred_at, CURRENT_TIMESTAMP);

    IF NEW.is_current THEN
        UPDATE declaration_qa.business_application_statuses
        SET is_current = FALSE,
            updated_at = CURRENT_TIMESTAMP
        WHERE apply_id = NEW.apply_id
          AND is_current = TRUE;
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_declaration_qa_prepare_business_status
    ON declaration_qa.business_application_statuses;
CREATE TRIGGER trg_declaration_qa_prepare_business_status
BEFORE INSERT ON declaration_qa.business_application_statuses
FOR EACH ROW
EXECUTE FUNCTION declaration_qa.prepare_business_application_status_event();

CREATE OR REPLACE FUNCTION declaration_qa.sync_business_application_current_status()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.is_current THEN
        UPDATE declaration_qa.business_applications
        SET status = NEW.status,
            current_node = NEW.current_node,
            node_name = NEW.node_name,
            updated_at = GREATEST(updated_at, NEW.occurred_at)
        WHERE id = NEW.apply_id;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_declaration_qa_sync_business_status
    ON declaration_qa.business_application_statuses;
CREATE TRIGGER trg_declaration_qa_sync_business_status
AFTER INSERT ON declaration_qa.business_application_statuses
FOR EACH ROW
EXECUTE FUNCTION declaration_qa.sync_business_application_current_status();


CREATE TABLE IF NOT EXISTS declaration_qa.business_reject_infos (
    id BIGSERIAL PRIMARY KEY,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    material_code VARCHAR(80) NOT NULL DEFAULT '',
    material_name VARCHAR(160) NOT NULL DEFAULT '',
    reject_code VARCHAR(80) NOT NULL DEFAULT '',
    reject_reason TEXT NOT NULL DEFAULT '',
    audit_node VARCHAR(80) NOT NULL DEFAULT '',
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_business_reject_current
    ON declaration_qa.business_reject_infos (apply_id)
    WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_reject_lookup
    ON declaration_qa.business_reject_infos (material_code, reject_code, audit_node);

CREATE TABLE IF NOT EXISTS declaration_qa.documents (
    id BIGSERIAL PRIMARY KEY,
    document_uid VARCHAR(120) NOT NULL DEFAULT '',
    knowledge_base_id BIGINT NOT NULL REFERENCES declaration_qa.knowledge_bases(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL DEFAULT '',
    content_type VARCHAR(120) NOT NULL DEFAULT 'text/plain',
    source_url TEXT NOT NULL DEFAULT '',
    raw_text TEXT NOT NULL,
    text_preview TEXT NOT NULL DEFAULT '',
    content_hash VARCHAR(80) NOT NULL DEFAULT '',
    audit_node VARCHAR(80) NOT NULL DEFAULT '',
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'indexing', 'published', 'failed', 'deprecated')),
    version_group_uid VARCHAR(120) NOT NULL DEFAULT '',
    version_no INTEGER NOT NULL DEFAULT 1 CHECK (version_no > 0),
    rollout_percent INTEGER NOT NULL DEFAULT 100 CHECK (rollout_percent >= 0 AND rollout_percent <= 100),
    change_note TEXT NOT NULL DEFAULT '',
    published_at TIMESTAMPTZ NULL,
    deprecated_at TIMESTAMPTZ NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_by BIGINT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
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
ALTER TABLE declaration_qa.documents
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_documents_policy_filter;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_documents_reject_filter;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_documents_effective_time;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_documents_audit_node;
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'documents' AND column_name = 'region_code'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.documents SET metadata = CASE WHEN metadata ? ''region_code'' THEN metadata ELSE jsonb_set(metadata, ''{region_code}'', to_jsonb(region_code::text), true) END WHERE region_code IS NOT NULL AND region_code <> ''''';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'documents' AND column_name = 'policy_version'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.documents SET metadata = CASE WHEN metadata ? ''policy_version'' THEN metadata ELSE jsonb_set(metadata, ''{policy_version}'', to_jsonb(policy_version::text), true) END WHERE policy_version IS NOT NULL AND policy_version <> ''''';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'documents' AND column_name = 'effective_start_time'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.documents SET metadata = CASE WHEN metadata ? ''effective_start_time'' THEN metadata ELSE jsonb_set(metadata, ''{effective_start_time}'', to_jsonb(effective_start_time::text), true) END WHERE effective_start_time IS NOT NULL';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'documents' AND column_name = 'effective_end_time'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.documents SET metadata = CASE WHEN metadata ? ''effective_end_time'' THEN metadata ELSE jsonb_set(metadata, ''{effective_end_time}'', to_jsonb(effective_end_time::text), true) END WHERE effective_end_time IS NOT NULL';
    END IF;
END $$;
ALTER TABLE declaration_qa.documents
    DROP COLUMN IF EXISTS region_code,
    DROP COLUMN IF EXISTS policy_version,
    DROP COLUMN IF EXISTS effective_start_time,
    DROP COLUMN IF EXISTS effective_end_time,
    DROP COLUMN IF EXISTS biz_type,
    DROP COLUMN IF EXISTS source_type,
    DROP COLUMN IF EXISTS material_code,
    DROP COLUMN IF EXISTS reject_code;
UPDATE declaration_qa.documents
SET metadata = metadata - 'biz_type' - 'source_type' - 'material_code' - 'reject_code'
    - 'bizType' - 'sourceType' - 'materialCode' - 'rejectCode'
    - 'theme_name' - 'themeName' - '主题分类' - '主题名称'
WHERE metadata ?| ARRAY['biz_type', 'source_type', 'material_code', 'reject_code', 'bizType', 'sourceType', 'materialCode', 'rejectCode', 'theme_name', 'themeName', '主题分类', '主题名称'];
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_documents_uid
    ON declaration_qa.documents (document_uid)
    WHERE document_uid <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_kb
    ON declaration_qa.documents (knowledge_base_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_version_group
    ON declaration_qa.documents (version_group_uid, version_no DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_documents_metadata_gin
    ON declaration_qa.documents USING GIN (metadata);

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
    source_url TEXT NOT NULL DEFAULT '',
    audit_node VARCHAR(80) NOT NULL DEFAULT '',
    status VARCHAR(20) NOT NULL DEFAULT 'published' CHECK (status IN ('draft', 'published', 'deprecated')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_document
    ON declaration_qa.chunks (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_parent
    ON declaration_qa.chunks (parent_chunk_id);
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_chunks_policy_filter;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_chunks_reject_filter;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_chunks_source_type;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_chunks_effective_time;
DROP INDEX IF EXISTS declaration_qa.ix_declaration_qa_chunks_audit_node;
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'chunks' AND column_name = 'region_code'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.chunks SET metadata = CASE WHEN metadata ? ''region_code'' THEN metadata ELSE jsonb_set(metadata, ''{region_code}'', to_jsonb(region_code::text), true) END WHERE region_code IS NOT NULL AND region_code <> ''''';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'chunks' AND column_name = 'policy_version'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.chunks SET metadata = CASE WHEN metadata ? ''policy_version'' THEN metadata ELSE jsonb_set(metadata, ''{policy_version}'', to_jsonb(policy_version::text), true) END WHERE policy_version IS NOT NULL AND policy_version <> ''''';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'chunks' AND column_name = 'effective_start_time'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.chunks SET metadata = CASE WHEN metadata ? ''effective_start_time'' THEN metadata ELSE jsonb_set(metadata, ''{effective_start_time}'', to_jsonb(effective_start_time::text), true) END WHERE effective_start_time IS NOT NULL';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'declaration_qa' AND table_name = 'chunks' AND column_name = 'effective_end_time'
    ) THEN
        EXECUTE 'UPDATE declaration_qa.chunks SET metadata = CASE WHEN metadata ? ''effective_end_time'' THEN metadata ELSE jsonb_set(metadata, ''{effective_end_time}'', to_jsonb(effective_end_time::text), true) END WHERE effective_end_time IS NOT NULL';
    END IF;
END $$;
ALTER TABLE declaration_qa.chunks
    DROP COLUMN IF EXISTS region_code,
    DROP COLUMN IF EXISTS policy_version,
    DROP COLUMN IF EXISTS effective_start_time,
    DROP COLUMN IF EXISTS effective_end_time,
    DROP COLUMN IF EXISTS biz_type,
    DROP COLUMN IF EXISTS source_type,
    DROP COLUMN IF EXISTS material_code,
    DROP COLUMN IF EXISTS reject_code;
UPDATE declaration_qa.chunks
SET metadata = metadata - 'biz_type' - 'source_type' - 'material_code' - 'reject_code'
    - 'bizType' - 'sourceType' - 'materialCode' - 'rejectCode'
    - 'theme_name' - 'themeName' - '主题分类' - '主题名称'
WHERE metadata ?| ARRAY['biz_type', 'source_type', 'material_code', 'reject_code', 'bizType', 'sourceType', 'materialCode', 'rejectCode', 'theme_name', 'themeName', '主题分类', '主题名称'];
CREATE INDEX IF NOT EXISTS ix_declaration_qa_chunks_metadata_gin
    ON declaration_qa.chunks USING GIN (metadata);

CREATE TABLE IF NOT EXISTS declaration_qa.assets (
    id BIGSERIAL PRIMARY KEY,
    asset_uid VARCHAR(120) NOT NULL UNIQUE,
    knowledge_base_id BIGINT NOT NULL REFERENCES declaration_qa.knowledge_bases(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES declaration_qa.documents(id) ON DELETE CASCADE,
    chunk_id BIGINT NULL REFERENCES declaration_qa.chunks(id) ON DELETE SET NULL,
    attachment_index INTEGER NOT NULL DEFAULT 0 CHECK (attachment_index >= 0),
    attachment_name VARCHAR(255) NOT NULL DEFAULT '',
    attachment_type VARCHAR(40) NOT NULL DEFAULT '',
    attachment_role VARCHAR(80) NOT NULL DEFAULT '',
    attachment_sha256 VARCHAR(80) NOT NULL DEFAULT '',
    asset_kind VARCHAR(80) NOT NULL DEFAULT '',
    modality VARCHAR(40) NOT NULL DEFAULT '',
    filename VARCHAR(255) NOT NULL DEFAULT '',
    content_type VARCHAR(120) NOT NULL DEFAULT '',
    file_path TEXT NOT NULL DEFAULT '',
    asset_url TEXT NOT NULL DEFAULT '',
    thumbnail_url TEXT NOT NULL DEFAULT '',
    page_no INTEGER NULL CHECK (page_no IS NULL OR page_no > 0),
    bbox JSONB NULL,
    ocr_text TEXT NOT NULL DEFAULT '',
    extraction_status VARCHAR(20) NOT NULL DEFAULT 'success' CHECK (extraction_status IN ('success', 'failed', 'skipped')),
    error_message TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_assets_document
    ON declaration_qa.assets (document_id, id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_assets_chunk
    ON declaration_qa.assets (chunk_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_assets_kb
    ON declaration_qa.assets (knowledge_base_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_assets_kind
    ON declaration_qa.assets (asset_kind, modality);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_assets_attachment
    ON declaration_qa.assets (attachment_sha256, attachment_index);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_assets_metadata_gin
    ON declaration_qa.assets USING GIN (metadata);

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

CREATE TABLE IF NOT EXISTS declaration_qa.agent_runs (
    id BIGSERIAL PRIMARY KEY,
    run_uid VARCHAR(80) NOT NULL UNIQUE,
    session_id BIGINT NOT NULL REFERENCES declaration_qa.sessions(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    iteration_count INTEGER NOT NULL DEFAULT 0,
    trace_id VARCHAR(120) NOT NULL DEFAULT '',
    answer_provider VARCHAR(120) NOT NULL DEFAULT '',
    idempotency_key VARCHAR(160) NOT NULL DEFAULT '',
    error_code VARCHAR(120) NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    prompt_version VARCHAR(80) NOT NULL DEFAULT '',
    tool_registry_version VARCHAR(80) NOT NULL DEFAULT '',
    model_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_agent_runs_session
    ON declaration_qa.agent_runs (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_agent_runs_user
    ON declaration_qa.agent_runs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_agent_runs_trace
    ON declaration_qa.agent_runs (trace_id)
    WHERE trace_id <> '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_agent_runs_idempotency
    ON declaration_qa.agent_runs (session_id, idempotency_key)
    WHERE idempotency_key <> '';

CREATE TABLE IF NOT EXISTS declaration_qa.messages (
    id BIGSERIAL PRIMARY KEY,
    message_uid VARCHAR(80) NOT NULL UNIQUE,
    session_id BIGINT NOT NULL REFERENCES declaration_qa.sessions(id) ON DELETE CASCADE,
    agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE CASCADE,
    sequence_no INTEGER NOT NULL DEFAULT 0,
    parent_message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    message_type VARCHAR(30) NOT NULL DEFAULT 'conversation',
    content TEXT NOT NULL,
    content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'completed'
        CHECK (status IN ('pending', 'completed', 'failed', 'cancelled')),
    visible BOOLEAN NOT NULL DEFAULT TRUE,
    model VARCHAR(160) NOT NULL DEFAULT '',
    provider VARCHAR(120) NOT NULL DEFAULT '',
    input_tokens INTEGER NULL CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER NULL CHECK (output_tokens IS NULL OR output_tokens >= 0),
    total_tokens INTEGER NULL CHECK (total_tokens IS NULL OR total_tokens >= 0),
    latency_ms BIGINT NULL CHECK (latency_ms IS NULL OR latency_ms >= 0),
    pii_level VARCHAR(30) NOT NULL DEFAULT 'none',
    retention_until TIMESTAMPTZ NULL,
    idempotency_key VARCHAR(160) NOT NULL DEFAULT '',
    intent VARCHAR(40) NOT NULL DEFAULT '',
    refused BOOLEAN NOT NULL DEFAULT FALSE,
    refuse_reason VARCHAR(120) NOT NULL DEFAULT '',
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    suggested_actions JSONB NOT NULL DEFAULT '[]'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE declaration_qa.messages
    ADD COLUMN IF NOT EXISTS message_uid VARCHAR(80),
    ADD COLUMN IF NOT EXISTS agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS sequence_no INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS parent_message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS message_type VARCHAR(30) NOT NULL DEFAULT 'conversation',
    ADD COLUMN IF NOT EXISTS content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'completed',
    ADD COLUMN IF NOT EXISTS visible BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS model VARCHAR(160) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS provider VARCHAR(120) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS input_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS output_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS total_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS latency_ms BIGINT NULL,
    ADD COLUMN IF NOT EXISTS pii_level VARCHAR(30) NOT NULL DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS retention_until TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(160) NOT NULL DEFAULT '';
UPDATE declaration_qa.messages
SET message_uid = 'legacy_message_' || id::text
WHERE message_uid IS NULL OR message_uid = '';
ALTER TABLE declaration_qa.messages
    ALTER COLUMN message_uid SET NOT NULL,
    DROP CONSTRAINT IF EXISTS messages_role_check,
    DROP CONSTRAINT IF EXISTS messages_status_check;
ALTER TABLE declaration_qa.messages
    ADD CONSTRAINT messages_role_check CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    ADD CONSTRAINT messages_status_check CHECK (status IN ('pending', 'completed', 'failed', 'cancelled'));
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_messages_uid
    ON declaration_qa.messages (message_uid);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_session
    ON declaration_qa.messages (session_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_messages_run_sequence
    ON declaration_qa.messages (agent_run_id, sequence_no)
    WHERE agent_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_run
    ON declaration_qa.messages (agent_run_id, sequence_no);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_parent
    ON declaration_qa.messages (parent_message_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_intent
    ON declaration_qa.messages (intent);

CREATE TABLE IF NOT EXISTS declaration_qa.tool_calls (
    id BIGSERIAL PRIMARY KEY,
    tool_call_uid VARCHAR(100) NOT NULL UNIQUE,
    agent_run_id BIGINT NOT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE CASCADE,
    assistant_message_id BIGINT NOT NULL REFERENCES declaration_qa.messages(id) ON DELETE CASCADE,
    result_message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL,
    external_call_id VARCHAR(160) NOT NULL DEFAULT '',
    sequence_no INTEGER NOT NULL,
    tool_name VARCHAR(160) NOT NULL,
    arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(80) NOT NULL DEFAULT 'pending',
    ok BOOLEAN NOT NULL DEFAULT FALSE,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT NOT NULL DEFAULT '',
    latency_ms BIGINT NULL CHECK (latency_ms IS NULL OR latency_ms >= 0),
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agent_run_id, sequence_no)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_tool_calls_run
    ON declaration_qa.tool_calls (agent_run_id, sequence_no);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_tool_calls_external
    ON declaration_qa.tool_calls (agent_run_id, external_call_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_tool_calls_name
    ON declaration_qa.tool_calls (tool_name, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.message_parts (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES declaration_qa.messages(id) ON DELETE CASCADE,
    part_index INTEGER NOT NULL,
    part_type VARCHAR(30) NOT NULL,
    text_content TEXT NOT NULL DEFAULT '',
    asset_uri TEXT NOT NULL DEFAULT '',
    mime_type VARCHAR(160) NOT NULL DEFAULT '',
    file_name VARCHAR(500) NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (message_id, part_index)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_parts_message
    ON declaration_qa.message_parts (message_id, part_index);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_parts_type
    ON declaration_qa.message_parts (part_type);

CREATE TABLE IF NOT EXISTS declaration_qa.message_sources (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES declaration_qa.messages(id) ON DELETE CASCADE,
    citation_no INTEGER NOT NULL,
    chunk_uid VARCHAR(160) NOT NULL DEFAULT '',
    document_uid VARCHAR(160) NOT NULL DEFAULT '',
    source_asset_id VARCHAR(200) NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    page_no INTEGER NULL,
    relevance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (message_id, citation_no)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_sources_message
    ON declaration_qa.message_sources (message_id, citation_no);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_sources_chunk
    ON declaration_qa.message_sources (chunk_uid)
    WHERE chunk_uid <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_sources_asset
    ON declaration_qa.message_sources (source_asset_id)
    WHERE source_asset_id <> '';

CREATE TABLE IF NOT EXISTS declaration_qa.audit_logs (
    id BIGSERIAL PRIMARY KEY,
    audit_uid VARCHAR(80) NOT NULL UNIQUE,
    session_id BIGINT NULL REFERENCES declaration_qa.sessions(id) ON DELETE SET NULL,
    agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE SET NULL,
    user_id BIGINT NOT NULL,
    query TEXT NOT NULL,
    normalized_query TEXT NOT NULL DEFAULT '',
    intent VARCHAR(40) NOT NULL,
    retrieval_chain VARCHAR(40) NOT NULL DEFAULT '',
    region_code VARCHAR(20) NOT NULL,
    biz_type VARCHAR(80) NULL,
    apply_id VARCHAR(80) NOT NULL DEFAULT '',
    business_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    agent_plan JSONB NOT NULL DEFAULT '{}'::jsonb,
    agent_query_plan JSONB NOT NULL DEFAULT '{}'::jsonb,
    tool_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    agent_iteration INTEGER NOT NULL DEFAULT 0,
    agent_answer_provider VARCHAR(120) NOT NULL DEFAULT '',
    agent_selected_item_code VARCHAR(160) NOT NULL DEFAULT '',
    agent_selected_application_id BIGINT NULL,
    business_status JSONB NOT NULL DEFAULT '{}'::jsonb,
    reject_info JSONB NOT NULL DEFAULT '{}'::jsonb,
    material_mapping JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_filter JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieved_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    rag_status JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence_ok BOOLEAN NOT NULL DEFAULT FALSE,
    evidence_decision JSONB NOT NULL DEFAULT '{}'::jsonb,
    refused BOOLEAN NOT NULL DEFAULT FALSE,
    refuse_reason VARCHAR(120) NOT NULL DEFAULT '',
    final_prompt TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    langfuse_trace_id VARCHAR(80) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS retrieval_chain VARCHAR(40) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE SET NULL;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS business_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_plan JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_query_plan JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS tool_results JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_iteration INTEGER NOT NULL DEFAULT 0;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_answer_provider VARCHAR(120) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_selected_item_code VARCHAR(160) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_selected_application_id BIGINT NULL;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS evidence_decision JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS langfuse_trace_id VARCHAR(80) NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_user
    ON declaration_qa.audit_logs (user_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_audit_logs_agent_run
    ON declaration_qa.audit_logs (agent_run_id)
    WHERE agent_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_intent
    ON declaration_qa.audit_logs (intent, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_chain
    ON declaration_qa.audit_logs (retrieval_chain, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_sources_gin
    ON declaration_qa.audit_logs USING GIN (retrieved_sources);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_filter_gin
    ON declaration_qa.audit_logs USING GIN (retrieval_filter);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_business_snapshot_gin
    ON declaration_qa.audit_logs USING GIN (business_snapshot);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_tool_results_gin
    ON declaration_qa.audit_logs USING GIN (tool_results);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_agent_query_plan_gin
    ON declaration_qa.audit_logs USING GIN (agent_query_plan);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_agent_item
    ON declaration_qa.audit_logs (agent_selected_item_code, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_langfuse_trace
    ON declaration_qa.audit_logs (langfuse_trace_id);

CREATE TABLE IF NOT EXISTS declaration_qa.feedback (
    id BIGSERIAL PRIMARY KEY,
    audit_log_id BIGINT NOT NULL REFERENCES declaration_qa.audit_logs(id) ON DELETE CASCADE,
    agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE SET NULL,
    message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL,
    user_id BIGINT NULL,
    rating VARCHAR(20) NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE declaration_qa.feedback
    ADD COLUMN IF NOT EXISTS agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_audit
    ON declaration_qa.feedback (audit_log_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_agent_run
    ON declaration_qa.feedback (agent_run_id)
    WHERE agent_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_message
    ON declaration_qa.feedback (message_id)
    WHERE message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_rating
    ON declaration_qa.feedback (rating, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.eval_datasets (
    id BIGSERIAL PRIMARY KEY,
    dataset_uid VARCHAR(80) NOT NULL UNIQUE,
    name VARCHAR(160) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    source VARCHAR(40) NOT NULL DEFAULT 'manual',
    parent_dataset_id BIGINT NULL REFERENCES declaration_qa.eval_datasets(id) ON DELETE SET NULL,
    knowledge_revision VARCHAR(160) NOT NULL DEFAULT '',
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
    reference_answer TEXT NOT NULL DEFAULT '',
    reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    case_type VARCHAR(30) NOT NULL DEFAULT 'rag',
    turns JSONB NOT NULL DEFAULT '[]'::jsonb,
    fixture_key VARCHAR(120) NOT NULL DEFAULT '',
    expected_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    expected_trajectory JSONB NOT NULL DEFAULT '{}'::jsonb,
    expected_outcome JSONB NOT NULL DEFAULT '{}'::jsonb,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
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
    status VARCHAR(20) NOT NULL DEFAULT 'running' CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    note TEXT NOT NULL DEFAULT '',
    top_k INTEGER NOT NULL DEFAULT 4 CHECK (top_k > 0),
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    baseline_run_id BIGINT NULL REFERENCES declaration_qa.eval_runs(id) ON DELETE SET NULL,
    trigger_source VARCHAR(40) NOT NULL DEFAULT 'manual',
    runtime_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    gate_result JSONB NOT NULL DEFAULT '{}'::jsonb,
    git_commit VARCHAR(80) NOT NULL DEFAULT '',
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
    reference_answer TEXT NOT NULL DEFAULT '',
    reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    retrieved_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer TEXT NOT NULL DEFAULT '',
    audit_uid VARCHAR(80) NOT NULL DEFAULT '',
    recall_hit BOOLEAN NULL,
    refusal_match BOOLEAN NULL,
    intent_match BOOLEAN NULL,
    keyword_hit BOOLEAN NULL,
    ragas_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ragas_status JSONB NOT NULL DEFAULT '{}'::jsonb,
    agent_run_uid VARCHAR(80) NOT NULL DEFAULT '',
    actual_trajectory JSONB NOT NULL DEFAULT '{}'::jsonb,
    rule_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    judge_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    failure_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    passed BOOLEAN NULL,
    input_tokens INTEGER NULL,
    output_tokens INTEGER NULL,
    total_tokens INTEGER NULL,
    latency_ms BIGINT NULL,
    estimated_cost NUMERIC(14, 6) NULL,
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

ALTER TABLE declaration_qa.eval_cases
    ADD COLUMN IF NOT EXISTS reference_answer TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE declaration_qa.eval_run_items
    ADD COLUMN IF NOT EXISTS reference_answer TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS retrieved_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS ragas_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS ragas_status JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE declaration_qa.eval_datasets
    ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'draft',
    ADD COLUMN IF NOT EXISTS source VARCHAR(40) NOT NULL DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS parent_dataset_id BIGINT NULL REFERENCES declaration_qa.eval_datasets(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS knowledge_revision VARCHAR(160) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.eval_cases
    ADD COLUMN IF NOT EXISTS case_type VARCHAR(30) NOT NULL DEFAULT 'rag',
    ADD COLUMN IF NOT EXISTS turns JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS fixture_key VARCHAR(120) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS expected_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS expected_trajectory JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS expected_outcome JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    ADD COLUMN IF NOT EXISTS enabled BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE declaration_qa.eval_runs
    ADD COLUMN IF NOT EXISTS baseline_run_id BIGINT NULL REFERENCES declaration_qa.eval_runs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS trigger_source VARCHAR(40) NOT NULL DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS runtime_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS gate_result JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS git_commit VARCHAR(80) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.eval_runs DROP CONSTRAINT IF EXISTS eval_runs_status_check;
ALTER TABLE declaration_qa.eval_runs
    ADD CONSTRAINT eval_runs_status_check
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled'));
ALTER TABLE declaration_qa.eval_run_items
    ADD COLUMN IF NOT EXISTS agent_run_uid VARCHAR(80) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS actual_trajectory JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS rule_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS judge_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS failure_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS passed BOOLEAN NULL,
    ADD COLUMN IF NOT EXISTS input_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS output_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS total_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS latency_ms BIGINT NULL,
    ADD COLUMN IF NOT EXISTS estimated_cost NUMERIC(14, 6) NULL;

CREATE TABLE IF NOT EXISTS declaration_qa.eval_case_revisions (
    id BIGSERIAL PRIMARY KEY,
    revision_uid VARCHAR(80) NOT NULL UNIQUE,
    case_id BIGINT NOT NULL REFERENCES declaration_qa.eval_cases(id) ON DELETE CASCADE,
    revision_no INTEGER NOT NULL,
    snapshot JSONB NOT NULL,
    change_note TEXT NOT NULL DEFAULT '',
    created_by BIGINT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (case_id, revision_no)
);
CREATE TABLE IF NOT EXISTS declaration_qa.eval_candidates (
    id BIGSERIAL PRIMARY KEY,
    candidate_uid VARCHAR(80) NOT NULL UNIQUE,
    audit_uid VARCHAR(80) NOT NULL DEFAULT '',
    agent_run_uid VARCHAR(80) NOT NULL DEFAULT '',
    trace_id VARCHAR(120) NOT NULL DEFAULT '',
    source VARCHAR(40) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'new',
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    sanitized_query TEXT NOT NULL DEFAULT '',
    sanitized_answer TEXT NOT NULL DEFAULT '',
    detection_signals JSONB NOT NULL DEFAULT '{}'::jsonb,
    trajectory_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    labels JSONB NOT NULL DEFAULT '[]'::jsonb,
    reviewer_comment TEXT NOT NULL DEFAULT '',
    reviewed_by BIGINT NULL,
    promoted_case_id BIGINT NULL REFERENCES declaration_qa.eval_cases(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMPTZ NULL
);
CREATE TABLE IF NOT EXISTS declaration_qa.eval_scores (
    id BIGSERIAL PRIMARY KEY,
    score_uid VARCHAR(80) NOT NULL UNIQUE,
    run_item_id BIGINT NOT NULL REFERENCES declaration_qa.eval_run_items(id) ON DELETE CASCADE,
    metric_name VARCHAR(120) NOT NULL,
    evaluator_type VARCHAR(40) NOT NULL,
    evaluator_version VARCHAR(80) NOT NULL DEFAULT '',
    score DOUBLE PRECISION NULL,
    label VARCHAR(40) NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    raw_result JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_datasets_status ON declaration_qa.eval_datasets (status, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_cases_type ON declaration_qa.eval_cases (dataset_id, case_type, enabled);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_runs_baseline ON declaration_qa.eval_runs (baseline_run_id) WHERE baseline_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_run_items_passed ON declaration_qa.eval_run_items (run_id, passed);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_candidates_status ON declaration_qa.eval_candidates (status, severity, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_candidates_audit ON declaration_qa.eval_candidates (audit_uid) WHERE audit_uid <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_scores_item ON declaration_qa.eval_scores (run_item_id, metric_name);

ALTER TABLE declaration_qa.eval_cases
    ADD COLUMN IF NOT EXISTS agent_retrieval_queries JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS source_trace_id VARCHAR(120) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE declaration_qa.eval_run_items
    ADD COLUMN IF NOT EXISTS agent_retrieval_queries JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS retrieval_ablation JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE declaration_qa.eval_candidates
    ADD COLUMN IF NOT EXISTS agent_retrieval_queries JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS online_scores JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_cases_source_trace
    ON declaration_qa.eval_cases (source_trace_id) WHERE source_trace_id <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_candidates_trace
    ON declaration_qa.eval_candidates (trace_id) WHERE trace_id <> '';

COMMENT ON SCHEMA declaration_qa IS '政务申报智能问答独立业务 schema';

COMMENT ON TABLE declaration_qa.knowledge_bases IS '知识库基础信息表';
COMMENT ON COLUMN declaration_qa.knowledge_bases.id IS '知识库自增主键';
COMMENT ON COLUMN declaration_qa.knowledge_bases.name IS '知识库名称';
COMMENT ON COLUMN declaration_qa.knowledge_bases.description IS '知识库说明';
COMMENT ON COLUMN declaration_qa.knowledge_bases.created_by IS '创建人用户 ID';
COMMENT ON COLUMN declaration_qa.knowledge_bases.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.knowledge_bases.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.business_applications IS '业务申请单主表，保存真实办件基础状态';
COMMENT ON COLUMN declaration_qa.business_applications.id IS '申请单自增主键';
COMMENT ON COLUMN declaration_qa.business_applications.apply_uid IS '外部申请单唯一标识';
COMMENT ON COLUMN declaration_qa.business_applications.user_id IS '申请人用户 ID';
COMMENT ON COLUMN declaration_qa.business_applications.item_code IS '政务服务事项编码';
COMMENT ON COLUMN declaration_qa.business_applications.item_name IS '政务服务事项名称';
COMMENT ON COLUMN declaration_qa.business_applications.biz_type IS '业务类型，仅用于真实工单上下文';
COMMENT ON COLUMN declaration_qa.business_applications.region_code IS '行政区划编码';
COMMENT ON COLUMN declaration_qa.business_applications.status IS '申请单当前状态';
COMMENT ON COLUMN declaration_qa.business_applications.current_node IS '当前审核节点编码';
COMMENT ON COLUMN declaration_qa.business_applications.node_name IS '当前审核节点名称';
COMMENT ON COLUMN declaration_qa.business_applications.submitted_at IS '申请提交时间';
COMMENT ON COLUMN declaration_qa.business_applications.completed_at IS '申请办结时间';
COMMENT ON COLUMN declaration_qa.business_applications.payload IS '业务系统原始扩展数据';
COMMENT ON COLUMN declaration_qa.business_applications.deleted_at IS '软删除时间';
COMMENT ON COLUMN declaration_qa.business_applications.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.business_applications.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.business_application_statuses IS '业务申请单状态流水表';
COMMENT ON COLUMN declaration_qa.business_application_statuses.id IS '状态记录自增主键';
COMMENT ON COLUMN declaration_qa.business_application_statuses.apply_id IS '关联 business_applications.id';
COMMENT ON COLUMN declaration_qa.business_application_statuses.sequence_no IS '申请内单调递增的状态事件序号';
COMMENT ON COLUMN declaration_qa.business_application_statuses.from_status IS '变更前状态编码';
COMMENT ON COLUMN declaration_qa.business_application_statuses.status IS '状态编码';
COMMENT ON COLUMN declaration_qa.business_application_statuses.current_node IS '审核节点编码';
COMMENT ON COLUMN declaration_qa.business_application_statuses.node_name IS '审核节点名称';
COMMENT ON COLUMN declaration_qa.business_application_statuses.reason_code IS '状态变更原因编码';
COMMENT ON COLUMN declaration_qa.business_application_statuses.reason IS '状态变更原因';
COMMENT ON COLUMN declaration_qa.business_application_statuses.material_code IS '状态事件关联材料编码';
COMMENT ON COLUMN declaration_qa.business_application_statuses.material_name IS '状态事件关联材料名称';
COMMENT ON COLUMN declaration_qa.business_application_statuses.source_event_id IS '上游业务系统事件唯一标识';
COMMENT ON COLUMN declaration_qa.business_application_statuses.is_current IS '是否当前有效状态';
COMMENT ON COLUMN declaration_qa.business_application_statuses.payload IS '状态原始扩展数据';
COMMENT ON COLUMN declaration_qa.business_application_statuses.occurred_at IS '状态在业务系统中的实际发生时间';
COMMENT ON COLUMN declaration_qa.business_application_statuses.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.business_application_statuses.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.business_reject_infos IS '业务申请单驳回和补正信息表';
COMMENT ON COLUMN declaration_qa.business_reject_infos.id IS '驳回记录自增主键';
COMMENT ON COLUMN declaration_qa.business_reject_infos.apply_id IS '关联 business_applications.id';
COMMENT ON COLUMN declaration_qa.business_reject_infos.material_code IS '业务系统材料编码';
COMMENT ON COLUMN declaration_qa.business_reject_infos.material_name IS '业务系统材料名称';
COMMENT ON COLUMN declaration_qa.business_reject_infos.reject_code IS '业务系统驳回原因编码';
COMMENT ON COLUMN declaration_qa.business_reject_infos.reject_reason IS '业务系统驳回原因说明';
COMMENT ON COLUMN declaration_qa.business_reject_infos.audit_node IS '驳回发生的审核节点';
COMMENT ON COLUMN declaration_qa.business_reject_infos.is_current IS '是否当前有效驳回记录';
COMMENT ON COLUMN declaration_qa.business_reject_infos.payload IS '驳回记录原始扩展数据';
COMMENT ON COLUMN declaration_qa.business_reject_infos.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.business_reject_infos.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.documents IS '知识文档表，保存 RAG 入库文档及发布版本信息';
COMMENT ON COLUMN declaration_qa.documents.id IS '文档自增主键';
COMMENT ON COLUMN declaration_qa.documents.document_uid IS '文档业务唯一标识';
COMMENT ON COLUMN declaration_qa.documents.knowledge_base_id IS '关联 knowledge_bases.id';
COMMENT ON COLUMN declaration_qa.documents.title IS '文档标题';
COMMENT ON COLUMN declaration_qa.documents.filename IS '上传文件名';
COMMENT ON COLUMN declaration_qa.documents.content_type IS '文件 MIME 类型';
COMMENT ON COLUMN declaration_qa.documents.source_url IS '来源链接或内部来源标识';
COMMENT ON COLUMN declaration_qa.documents.raw_text IS '解析后的完整文本';
COMMENT ON COLUMN declaration_qa.documents.text_preview IS '文本摘要预览';
COMMENT ON COLUMN declaration_qa.documents.content_hash IS '文档内容哈希';
COMMENT ON COLUMN declaration_qa.documents.audit_node IS '可选审核节点，仅用于展示、审计和溯源';
COMMENT ON COLUMN declaration_qa.documents.status IS '文档状态：draft/indexing/published/failed/deprecated';
COMMENT ON COLUMN declaration_qa.documents.version_group_uid IS '版本组唯一标识';
COMMENT ON COLUMN declaration_qa.documents.version_no IS '版本号';
COMMENT ON COLUMN declaration_qa.documents.rollout_percent IS '灰度发布比例';
COMMENT ON COLUMN declaration_qa.documents.change_note IS '版本变更说明';
COMMENT ON COLUMN declaration_qa.documents.published_at IS '发布时间';
COMMENT ON COLUMN declaration_qa.documents.deprecated_at IS '下线时间';
COMMENT ON COLUMN declaration_qa.documents.chunk_count IS '已生成分块数量';
COMMENT ON COLUMN declaration_qa.documents.metadata IS '文档扩展元数据，仅用于展示、审计和溯源，不参与 RAG 检索过滤';
COMMENT ON COLUMN declaration_qa.documents.error_message IS '索引或解析失败原因';
COMMENT ON COLUMN declaration_qa.documents.created_by IS '创建人用户 ID';
COMMENT ON COLUMN declaration_qa.documents.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.documents.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.chunks IS '知识文档分块表，保存父子分块和检索元数据';
COMMENT ON COLUMN declaration_qa.chunks.id IS '分块自增主键';
COMMENT ON COLUMN declaration_qa.chunks.document_id IS '关联 documents.id';
COMMENT ON COLUMN declaration_qa.chunks.parent_chunk_id IS '父分块 ID，子分块用于展开上下文';
COMMENT ON COLUMN declaration_qa.chunks.chunk_uid IS '分块业务唯一标识';
COMMENT ON COLUMN declaration_qa.chunks.chunk_type IS '分块类型：parent/child';
COMMENT ON COLUMN declaration_qa.chunks.chunk_index IS '分块顺序号';
COMMENT ON COLUMN declaration_qa.chunks.content IS '分块文本内容';
COMMENT ON COLUMN declaration_qa.chunks.content_hash IS '分块内容哈希';
COMMENT ON COLUMN declaration_qa.chunks.token_count IS '估算 token 数';
COMMENT ON COLUMN declaration_qa.chunks.embedding_id IS '向量索引中的 embedding 标识';
COMMENT ON COLUMN declaration_qa.chunks.vector_collection IS '向量集合名称';
COMMENT ON COLUMN declaration_qa.chunks.source_url IS '来源链接或内部来源标识';
COMMENT ON COLUMN declaration_qa.chunks.audit_node IS '可选审核节点，仅用于展示、审计和溯源';
COMMENT ON COLUMN declaration_qa.chunks.status IS '分块状态：draft/published/deprecated';
COMMENT ON COLUMN declaration_qa.chunks.metadata IS '分块扩展元数据，不包含已移除的 biz_type/source_type/material_code/reject_code/theme_name，且不参与 RAG 检索过滤';
COMMENT ON COLUMN declaration_qa.chunks.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.chunks.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.assets IS '知识文档附件资产表，保存 PDF 页面截图、DOCX 内嵌图片和图片附件等可展示资产';
COMMENT ON COLUMN declaration_qa.assets.id IS '资产自增主键';
COMMENT ON COLUMN declaration_qa.assets.asset_uid IS '资产业务唯一标识，对应 chunk metadata 中的 asset_id';
COMMENT ON COLUMN declaration_qa.assets.knowledge_base_id IS '关联 knowledge_bases.id';
COMMENT ON COLUMN declaration_qa.assets.document_id IS '关联 documents.id';
COMMENT ON COLUMN declaration_qa.assets.chunk_id IS '关联图片资产 chunk，可为空';
COMMENT ON COLUMN declaration_qa.assets.attachment_index IS '来源附件在文档中的顺序';
COMMENT ON COLUMN declaration_qa.assets.attachment_name IS '来源附件名称';
COMMENT ON COLUMN declaration_qa.assets.attachment_type IS '来源附件扩展名';
COMMENT ON COLUMN declaration_qa.assets.attachment_role IS '来源附件角色：空表、样表、流程图或其他附件';
COMMENT ON COLUMN declaration_qa.assets.attachment_sha256 IS '来源附件内容 SHA256';
COMMENT ON COLUMN declaration_qa.assets.asset_kind IS '资产类型：pdf_page_image/docx_embedded_image/original_image';
COMMENT ON COLUMN declaration_qa.assets.modality IS '资产模态：image/text/table 等';
COMMENT ON COLUMN declaration_qa.assets.filename IS '资产文件名';
COMMENT ON COLUMN declaration_qa.assets.content_type IS '资产 MIME 类型';
COMMENT ON COLUMN declaration_qa.assets.file_path IS '资产本地落盘路径';
COMMENT ON COLUMN declaration_qa.assets.asset_url IS '资产 API 访问地址';
COMMENT ON COLUMN declaration_qa.assets.thumbnail_url IS '缩略图 API 访问地址';
COMMENT ON COLUMN declaration_qa.assets.page_no IS 'PDF 页面资产对应页码';
COMMENT ON COLUMN declaration_qa.assets.bbox IS '资产在页面中的坐标区域';
COMMENT ON COLUMN declaration_qa.assets.ocr_text IS '资产 OCR 或页面关联文本';
COMMENT ON COLUMN declaration_qa.assets.extraction_status IS '资产抽取状态';
COMMENT ON COLUMN declaration_qa.assets.error_message IS '资产抽取失败原因';
COMMENT ON COLUMN declaration_qa.assets.metadata IS '资产扩展元数据';
COMMENT ON COLUMN declaration_qa.assets.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.assets.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.index_jobs IS '知识索引异步任务表';
COMMENT ON COLUMN declaration_qa.index_jobs.id IS '索引任务自增主键';
COMMENT ON COLUMN declaration_qa.index_jobs.job_uid IS '索引任务唯一标识';
COMMENT ON COLUMN declaration_qa.index_jobs.knowledge_base_id IS '关联 knowledge_bases.id';
COMMENT ON COLUMN declaration_qa.index_jobs.document_id IS '关联 documents.id';
COMMENT ON COLUMN declaration_qa.index_jobs.job_type IS '任务类型：document_index/knowledge_base_reindex/document_delete';
COMMENT ON COLUMN declaration_qa.index_jobs.status IS '任务状态：pending/running/succeeded/failed/cancelled';
COMMENT ON COLUMN declaration_qa.index_jobs.progress IS '任务进度百分比';
COMMENT ON COLUMN declaration_qa.index_jobs.message IS '任务进度消息';
COMMENT ON COLUMN declaration_qa.index_jobs.error_message IS '任务失败原因';
COMMENT ON COLUMN declaration_qa.index_jobs.payload IS '任务请求和执行结果载荷';
COMMENT ON COLUMN declaration_qa.index_jobs.started_at IS '任务开始时间';
COMMENT ON COLUMN declaration_qa.index_jobs.completed_at IS '任务完成时间';
COMMENT ON COLUMN declaration_qa.index_jobs.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.index_jobs.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.sessions IS '问答会话表';
COMMENT ON COLUMN declaration_qa.sessions.id IS '会话自增主键';
COMMENT ON COLUMN declaration_qa.sessions.session_uid IS '会话唯一标识';
COMMENT ON COLUMN declaration_qa.sessions.user_id IS '用户 ID';
COMMENT ON COLUMN declaration_qa.sessions.region_code IS '会话地区编码';
COMMENT ON COLUMN declaration_qa.sessions.biz_type IS '会话业务类型，仅用于业务上下文和前端展示';
COMMENT ON COLUMN declaration_qa.sessions.apply_id IS '会话绑定的申请单 ID';
COMMENT ON COLUMN declaration_qa.sessions.title IS '会话标题';
COMMENT ON COLUMN declaration_qa.sessions.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.sessions.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.messages IS '问答消息表';
COMMENT ON COLUMN declaration_qa.messages.id IS '消息自增主键';
COMMENT ON COLUMN declaration_qa.messages.session_id IS '关联 sessions.id';
COMMENT ON COLUMN declaration_qa.messages.role IS '消息角色：user/assistant/system/tool';
COMMENT ON COLUMN declaration_qa.messages.content IS '消息内容';
COMMENT ON COLUMN declaration_qa.messages.intent IS '识别出的问答意图';
COMMENT ON COLUMN declaration_qa.messages.refused IS '是否拒答';
COMMENT ON COLUMN declaration_qa.messages.refuse_reason IS '拒答原因编码';
COMMENT ON COLUMN declaration_qa.messages.sources IS '回答引用来源列表';
COMMENT ON COLUMN declaration_qa.messages.suggested_actions IS '建议操作列表';
COMMENT ON COLUMN declaration_qa.messages.meta IS '消息扩展元数据';
COMMENT ON COLUMN declaration_qa.messages.created_at IS '创建时间';

COMMENT ON TABLE declaration_qa.audit_logs IS '问答审计日志表，记录业务快照、检索、证据和最终回答';
COMMENT ON COLUMN declaration_qa.audit_logs.id IS '审计日志自增主键';
COMMENT ON COLUMN declaration_qa.audit_logs.audit_uid IS '审计日志唯一标识';
COMMENT ON COLUMN declaration_qa.audit_logs.session_id IS '关联 sessions.id';
COMMENT ON COLUMN declaration_qa.audit_logs.user_id IS '用户 ID';
COMMENT ON COLUMN declaration_qa.audit_logs.query IS '用户原始问题';
COMMENT ON COLUMN declaration_qa.audit_logs.normalized_query IS '归一化后的问题';
COMMENT ON COLUMN declaration_qa.audit_logs.intent IS '识别出的问答意图';
COMMENT ON COLUMN declaration_qa.audit_logs.retrieval_chain IS '回答链路：agentic_rag/knowledge_retrieval';
COMMENT ON COLUMN declaration_qa.audit_logs.region_code IS '行政区划编码';
COMMENT ON COLUMN declaration_qa.audit_logs.biz_type IS '业务类型，仅用于审计业务上下文';
COMMENT ON COLUMN declaration_qa.audit_logs.apply_id IS '申请单 ID';
COMMENT ON COLUMN declaration_qa.audit_logs.business_snapshot IS '真实工单业务快照';
COMMENT ON COLUMN declaration_qa.audit_logs.agent_plan IS 'Agent 最近一次决策及工具调用计划';
COMMENT ON COLUMN declaration_qa.audit_logs.agent_query_plan IS 'Agent 请求和服务端实际执行的知识查询计划';
COMMENT ON COLUMN declaration_qa.audit_logs.tool_results IS 'Tool Registry 的完整结构化执行结果';
COMMENT ON COLUMN declaration_qa.audit_logs.agent_iteration IS 'Agent 有界循环实际迭代次数';
COMMENT ON COLUMN declaration_qa.audit_logs.agent_answer_provider IS '最终答案提供方，正常为 Tool-Calling Agent';
COMMENT ON COLUMN declaration_qa.audit_logs.agent_selected_item_code IS 'Agent 自主选择的事项编码';
COMMENT ON COLUMN declaration_qa.audit_logs.agent_selected_application_id IS 'Agent 自主选择的 application 主键';
COMMENT ON COLUMN declaration_qa.audit_logs.business_status IS '真实工单状态查询结果';
COMMENT ON COLUMN declaration_qa.audit_logs.reject_info IS '真实工单驳回或补正查询结果';
COMMENT ON COLUMN declaration_qa.audit_logs.material_mapping IS '材料补正映射结果';
COMMENT ON COLUMN declaration_qa.audit_logs.retrieval_filter IS 'RAG 实际检索过滤条件';
COMMENT ON COLUMN declaration_qa.audit_logs.retrieved_sources IS 'RAG 召回和精排后的来源';
COMMENT ON COLUMN declaration_qa.audit_logs.rag_status IS 'RAG 检索诊断状态';
COMMENT ON COLUMN declaration_qa.audit_logs.evidence_ok IS '证据校验是否通过';
COMMENT ON COLUMN declaration_qa.audit_logs.evidence_decision IS '证据校验决策详情';
COMMENT ON COLUMN declaration_qa.audit_logs.refused IS '是否拒答';
COMMENT ON COLUMN declaration_qa.audit_logs.refuse_reason IS '拒答原因编码';
COMMENT ON COLUMN declaration_qa.audit_logs.final_prompt IS '最终生成提示词';
COMMENT ON COLUMN declaration_qa.audit_logs.answer IS '最终回答文本';
COMMENT ON COLUMN declaration_qa.audit_logs.events IS '工作流事件流水';
COMMENT ON COLUMN declaration_qa.audit_logs.langfuse_trace_id IS 'Langfuse trace id，用于用户反馈反写 score';
COMMENT ON COLUMN declaration_qa.audit_logs.created_at IS '创建时间';

COMMENT ON TABLE declaration_qa.feedback IS '用户反馈表';
COMMENT ON COLUMN declaration_qa.feedback.id IS '反馈自增主键';
COMMENT ON COLUMN declaration_qa.feedback.audit_log_id IS '关联 audit_logs.id';
COMMENT ON COLUMN declaration_qa.feedback.user_id IS '反馈用户 ID';
COMMENT ON COLUMN declaration_qa.feedback.rating IS '反馈评分：up/down';
COMMENT ON COLUMN declaration_qa.feedback.comment IS '反馈备注';
COMMENT ON COLUMN declaration_qa.feedback.created_at IS '创建时间';

COMMENT ON TABLE declaration_qa.eval_datasets IS 'RAG 评测数据集表';
COMMENT ON COLUMN declaration_qa.eval_datasets.id IS '评测数据集自增主键';
COMMENT ON COLUMN declaration_qa.eval_datasets.dataset_uid IS '评测数据集唯一标识';
COMMENT ON COLUMN declaration_qa.eval_datasets.name IS '评测数据集名称';
COMMENT ON COLUMN declaration_qa.eval_datasets.description IS '评测数据集说明';
COMMENT ON COLUMN declaration_qa.eval_datasets.created_by IS '创建人用户 ID';
COMMENT ON COLUMN declaration_qa.eval_datasets.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.eval_datasets.updated_at IS '更新时间';

COMMENT ON TABLE declaration_qa.eval_cases IS 'RAG 评测用例表';
COMMENT ON COLUMN declaration_qa.eval_cases.id IS '评测用例自增主键';
COMMENT ON COLUMN declaration_qa.eval_cases.case_uid IS '评测用例唯一标识';
COMMENT ON COLUMN declaration_qa.eval_cases.dataset_id IS '关联 eval_datasets.id';
COMMENT ON COLUMN declaration_qa.eval_cases.query IS '评测问题';
COMMENT ON COLUMN declaration_qa.eval_cases.user_id IS '评测用户 ID';
COMMENT ON COLUMN declaration_qa.eval_cases.region_code IS '评测地区编码';
COMMENT ON COLUMN declaration_qa.eval_cases.policy_version IS '评测政策版本';
COMMENT ON COLUMN declaration_qa.eval_cases.biz_type IS '评测业务类型，仅用于请求上下文';
COMMENT ON COLUMN declaration_qa.eval_cases.apply_id IS '评测申请单 ID';
COMMENT ON COLUMN declaration_qa.eval_cases.apply_context IS '评测申请上下文';
COMMENT ON COLUMN declaration_qa.eval_cases.expected_intent IS '期望意图';
COMMENT ON COLUMN declaration_qa.eval_cases.expected_refused IS '期望是否拒答';
COMMENT ON COLUMN declaration_qa.eval_cases.expected_doc_ids IS '期望命中的文档 ID 列表';
COMMENT ON COLUMN declaration_qa.eval_cases.expected_answer_keywords IS '期望回答关键词';
COMMENT ON COLUMN declaration_qa.eval_cases.tags IS '评测标签';
COMMENT ON COLUMN declaration_qa.eval_cases.created_at IS '创建时间';

COMMENT ON TABLE declaration_qa.eval_runs IS 'RAG 评测运行表';
COMMENT ON COLUMN declaration_qa.eval_runs.id IS '评测运行自增主键';
COMMENT ON COLUMN declaration_qa.eval_runs.run_uid IS '评测运行唯一标识';
COMMENT ON COLUMN declaration_qa.eval_runs.dataset_id IS '关联 eval_datasets.id';
COMMENT ON COLUMN declaration_qa.eval_runs.status IS '运行状态：running/succeeded/failed';
COMMENT ON COLUMN declaration_qa.eval_runs.note IS '运行备注';
COMMENT ON COLUMN declaration_qa.eval_runs.top_k IS '召回评测 top_k 参数';
COMMENT ON COLUMN declaration_qa.eval_runs.metrics IS '运行聚合指标';
COMMENT ON COLUMN declaration_qa.eval_runs.error_message IS '运行失败原因';
COMMENT ON COLUMN declaration_qa.eval_runs.created_at IS '创建时间';
COMMENT ON COLUMN declaration_qa.eval_runs.completed_at IS '完成时间';

COMMENT ON TABLE declaration_qa.eval_run_items IS 'RAG 评测运行明细表';
COMMENT ON COLUMN declaration_qa.eval_run_items.id IS '运行明细自增主键';
COMMENT ON COLUMN declaration_qa.eval_run_items.item_uid IS '运行明细唯一标识';
COMMENT ON COLUMN declaration_qa.eval_run_items.run_id IS '关联 eval_runs.id';
COMMENT ON COLUMN declaration_qa.eval_run_items.case_id IS '关联 eval_cases.id';
COMMENT ON COLUMN declaration_qa.eval_run_items.query IS '评测问题';
COMMENT ON COLUMN declaration_qa.eval_run_items.expected_intent IS '期望意图';
COMMENT ON COLUMN declaration_qa.eval_run_items.actual_intent IS '实际意图';
COMMENT ON COLUMN declaration_qa.eval_run_items.expected_refused IS '期望是否拒答';
COMMENT ON COLUMN declaration_qa.eval_run_items.actual_refused IS '实际是否拒答';
COMMENT ON COLUMN declaration_qa.eval_run_items.expected_doc_ids IS '期望命中的文档 ID 列表';
COMMENT ON COLUMN declaration_qa.eval_run_items.retrieved_doc_ids IS '实际召回的文档 ID 列表';
COMMENT ON COLUMN declaration_qa.eval_run_items.expected_answer_keywords IS '期望回答关键词';
COMMENT ON COLUMN declaration_qa.eval_run_items.answer IS '实际回答文本';
COMMENT ON COLUMN declaration_qa.eval_run_items.audit_uid IS '关联审计日志唯一标识';
COMMENT ON COLUMN declaration_qa.eval_run_items.recall_hit IS '召回是否命中';
COMMENT ON COLUMN declaration_qa.eval_run_items.refusal_match IS '拒答结果是否匹配';
COMMENT ON COLUMN declaration_qa.eval_run_items.intent_match IS '意图识别是否匹配';
COMMENT ON COLUMN declaration_qa.eval_run_items.keyword_hit IS '回答关键词是否命中';
COMMENT ON COLUMN declaration_qa.eval_run_items.annotation_label IS '人工标注标签';
COMMENT ON COLUMN declaration_qa.eval_run_items.annotation_comment IS '人工标注备注';
COMMENT ON COLUMN declaration_qa.eval_run_items.annotated_at IS '人工标注时间';
COMMENT ON COLUMN declaration_qa.eval_run_items.created_at IS '创建时间';

COMMENT ON TABLE declaration_qa.agent_runs IS '一次用户请求对应的 Agent 执行记录';
COMMENT ON TABLE declaration_qa.tool_calls IS 'Agent 工具调用及其执行生命周期';
COMMENT ON TABLE declaration_qa.message_parts IS '消息的文本、图片、文件和 OCR 多模态组成部分';
COMMENT ON TABLE declaration_qa.message_sources IS '最终回答消息与知识证据的引用关系';
COMMENT ON COLUMN declaration_qa.messages.message_uid IS '消息稳定唯一标识';
COMMENT ON COLUMN declaration_qa.messages.agent_run_id IS '关联 agent_runs.id';
COMMENT ON COLUMN declaration_qa.messages.sequence_no IS '消息在单次 Agent run 内的严格顺序';
COMMENT ON COLUMN declaration_qa.messages.parent_message_id IS '父消息，用于表达工具结果和并行分支关系';
COMMENT ON COLUMN declaration_qa.messages.message_type IS '消息类型：system/input/model_input/tool_call/tool_result/output';
COMMENT ON COLUMN declaration_qa.messages.content_json IS '工具调用等结构化消息内容';
COMMENT ON COLUMN declaration_qa.messages.visible IS '是否出现在普通会话历史中';

COMMIT;

-- ============================================================================
-- 来源：014_multiturn_form_filling.sql
-- ============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS declaration_qa.business_item_form_schemas (
    id BIGSERIAL PRIMARY KEY,
    schema_uid VARCHAR(160) NOT NULL UNIQUE,
    item_code VARCHAR(160) NOT NULL,
    item_name VARCHAR(255) NOT NULL,
    biz_type VARCHAR(80) NOT NULL DEFAULT 'application',
    region_code VARCHAR(20) NOT NULL DEFAULT '',
    version_no INTEGER NOT NULL DEFAULT 1 CHECK (version_no > 0),
    status VARCHAR(40) NOT NULL DEFAULT 'draft',
    fields JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(fields) = 'array'),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (item_code, region_code, version_no)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_item_form_schemas_lookup
    ON declaration_qa.business_item_form_schemas (item_code, region_code, status, version_no DESC);

ALTER TABLE declaration_qa.business_applications
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(120) NOT NULL DEFAULT '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_applications_idempotency
    ON declaration_qa.business_applications (user_id, idempotency_key)
    WHERE idempotency_key <> '';

INSERT INTO declaration_qa.business_item_form_schemas (
    schema_uid, item_code, item_name, biz_type, region_code, version_no, status, fields, metadata
) VALUES
(
    'residence-permit-v1', 'ITEM-RESIDENCE-PERMIT', '居住证办理', 'residence_permit', '', 1, 'published',
    '[{"name":"applicant_name","label":"申请人姓名","type":"text","required":true,"max_length":80,"aliases":["姓名","申请人"]},{"name":"id_number","label":"证件号码","type":"id_number","required":true,"aliases":["身份证号","身份证号码"]},{"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},{"name":"residence_address","label":"现居住地址","type":"text","required":true,"max_length":300,"aliases":["住址","居住地址"]},{"name":"residence_proof_no","label":"居住证明编号","type":"text","required":true,"max_length":100,"aliases":["证明编号"]}]'::jsonb,
    jsonb_build_object('source', 'multiturn_form_filling_seed')
),
(
    'health-certificate-v1', 'ITEM-HEALTH-CERTIFICATE', '健康证办理', 'health_certificate', '', 1, 'published',
    '[{"name":"applicant_name","label":"申请人姓名","type":"text","required":true,"max_length":80,"aliases":["姓名","申请人"]},{"name":"id_number","label":"证件号码","type":"id_number","required":true,"aliases":["身份证号","身份证号码"]},{"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},{"name":"employer_name","label":"从业单位","type":"text","required":true,"max_length":200,"aliases":["单位","工作单位"]},{"name":"occupation_type","label":"从业类别","type":"select","required":true,"options":["食品","公共场所","饮用水","化妆品","其他"],"aliases":["职业类别","行业"]}]'::jsonb,
    jsonb_build_object('source', 'multiturn_form_filling_seed')
),
(
    'water-permit-v1', '11340000002985923L200011900100001', '取水许可', 'water_permit', '', 1, 'published',
    '[{"name":"applicant_name","label":"申请单位或个人","type":"text","required":true,"max_length":200,"aliases":["申请人","单位名称"]},{"name":"credit_code","label":"统一社会信用代码或证件号码","type":"text","required":true,"max_length":40,"aliases":["统一社会信用代码","证件号码"]},{"name":"contact_name","label":"联系人","type":"text","required":true,"max_length":80,"aliases":["经办人"]},{"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话"]},{"name":"water_source","label":"取水水源","type":"text","required":true,"max_length":200,"aliases":["水源"]},{"name":"annual_water_volume","label":"年取水量（立方米）","type":"number","required":true,"aliases":["年取水量","取水量"]},{"name":"water_use","label":"取水用途","type":"multiselect","required":true,"options":["制水供水","原水供水","生活用水","工业用水","农业用水","其他"],"aliases":["用途"]}]'::jsonb,
    jsonb_build_object('source', 'multiturn_form_filling_seed')
)
ON CONFLICT (schema_uid) DO UPDATE SET
    item_name = EXCLUDED.item_name,
    biz_type = EXCLUDED.biz_type,
    status = EXCLUDED.status,
    fields = EXCLUDED.fields,
    metadata = EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

COMMIT;


-- ============================================================================
-- 来源：004_declaration_qa_seed_business_real10.sql
-- ============================================================================

-- 模拟业务工单数据生成自：
-- materials/curated/declaration_qa/2026-06-15-ahzwfw-real-10
--
-- 此种子数据特意限定为 apply_uid LIKE 'AHZWFW-REAL10-%'，因此可重复运行，
-- 且不会影响真实业务工单。

BEGIN;

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-REAL10-%'
)
DELETE FROM declaration_qa.business_reject_infos
WHERE apply_id IN (SELECT id FROM target_apps);

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-REAL10-%'
)
DELETE FROM declaration_qa.business_application_statuses
WHERE apply_id IN (SELECT id FROM target_apps);

WITH seed (
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-REAL10-001',
        1,
        'engineering_survey_design_master_selection',
        '340000',
        'in_review',
        'REVIEW',
        '审查',
        '2026-06-02 09:20:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '安徽省工程勘察设计大师评选',
            'item_code', '11340000MB0T3248X7234101703500001',
            'source_file', 'engineering_survey_design_master_selection.md',
            'materials', jsonb_build_array('申报材料意见函', '安徽省工程勘察设计大师申报表', '勘察设计大师申报附件材料'),
            'legal_limit', '20个工作日',
            'promise_limit', '20个工作日',
            'handle_form', '网上办理,窗口办理,快递申请'
        )
    ),
    (
        'AHZWFW-REAL10-002',
        1,
        'second_level_constructor_additional_registration',
        '340000',
        'accepted',
        'ACCEPT',
        '受理',
        '2026-06-03 10:10:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级建造师增项注册',
            'item_code', '11340000MB0T3248X7200011705500202',
            'source_file', 'second_level_constructor_additional_registration.md',
            'materials', jsonb_build_array('二级建造师增项注册表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理,自助端办理'
        )
    ),
    (
        'AHZWFW-REAL10-003',
        1,
        'second_level_constructor_initial_registration',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-04 14:35:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级建造师初始注册',
            'item_code', '11340000MB0T3248X7200011705500205',
            'source_file', 'second_level_constructor_initial_registration.md',
            'materials', jsonb_build_array('二级建造师初始注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-004',
        1,
        'second_level_constructor_re_registration',
        '340000',
        'in_review',
        'REVIEW',
        '审查',
        '2026-06-05 09:05:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级建造师重新注册',
            'item_code', '11340000MB0T3248X7200011705500209',
            'source_file', 'second_level_constructor_re_registration.md',
            'materials', jsonb_build_array('二级建造师重新注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-005',
        1,
        'second_level_constructor_renewal_registration',
        '340000',
        'completed',
        'FINISH',
        '办结',
        '2026-05-28 11:25:00+08',
        '2026-06-06 16:10:00+08',
        jsonb_build_object(
            'item_name', '二级建造师延续注册',
            'item_code', '11340000MB0T3248X7200011705500206',
            'source_file', 'second_level_constructor_renewal_registration.md',
            'materials', jsonb_build_array('二级建造师延续注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-006',
        1,
        'second_level_cost_engineer_cancellation_registration',
        '340000',
        'pre_accept',
        'PRE_ACCEPT',
        '网上预审',
        '2026-06-07 15:40:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师注销注册',
            'item_code', '11340000MB0T3248X7200011705600003',
            'source_file', 'second_level_cost_engineer_cancellation_registration.md',
            'materials', jsonb_build_array('二级造价工程师注销注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-007',
        1,
        'second_level_cost_engineer_change_registration',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-08 09:55:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师变更注册',
            'item_code', '11340000MB0T3248X7200011705600005',
            'source_file', 'second_level_cost_engineer_change_registration.md',
            'materials', jsonb_build_array('二级造价工程师变更注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-008',
        1,
        'second_level_cost_engineer_initial_registration',
        '340000',
        'in_decision',
        'DECISION',
        '决定',
        '2026-06-09 13:50:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师初始注册',
            'item_code', '11340000MB0T3248X7200011705600002',
            'source_file', 'second_level_cost_engineer_initial_registration.md',
            'materials', jsonb_build_array('二级造价工程师初始注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-009',
        1,
        'second_level_cost_engineer_renewal_registration',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-10 10:05:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师延续注册',
            'item_code', '11340000MB0T3248X7200011705600004',
            'source_file', 'second_level_cost_engineer_renewal_registration.md',
            'materials', jsonb_build_array('二级造价工程师延续注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-010',
        1,
        'second_level_structural_engineer_initial_registration',
        '340000',
        'accepted',
        'ACCEPT',
        '受理',
        '2026-06-11 16:25:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级结构工程师初始注册',
            'item_code', '11340000MB0T3248X7200011703900206',
            'source_file', 'second_level_structural_engineer_initial_registration.md',
            'materials', jsonb_build_array('二级注册结构工程师初始注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    )
)
INSERT INTO declaration_qa.business_applications (
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    payload,
    deleted_at,
    created_at,
    updated_at
)
SELECT
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at::timestamptz,
    completed_at::timestamptz,
    payload,
    NULL::timestamptz,
    submitted_at::timestamptz,
    COALESCE(completed_at::timestamptz, submitted_at::timestamptz + INTERVAL '2 days')
FROM seed
ON CONFLICT (apply_uid) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    biz_type = EXCLUDED.biz_type,
    region_code = EXCLUDED.region_code,
    status = EXCLUDED.status,
    current_node = EXCLUDED.current_node,
    node_name = EXCLUDED.node_name,
    submitted_at = EXCLUDED.submitted_at,
    completed_at = EXCLUDED.completed_at,
    payload = EXCLUDED.payload,
    deleted_at = NULL,
    updated_at = EXCLUDED.updated_at;

WITH seed_status (
    apply_uid,
    status,
    current_node,
    node_name,
    is_current,
    created_at,
    updated_at,
    payload
) AS (
    VALUES
    ('AHZWFW-REAL10-001', 'in_review', 'REVIEW', '审查', TRUE, '2026-06-04 09:00:00+08'::timestamptz, '2026-06-12 10:20:00+08'::timestamptz, jsonb_build_object('step', '审查', 'message', '已进入主管部门审查环节，正在核验申报材料和业绩证明。')),
    ('AHZWFW-REAL10-002', 'accepted', 'ACCEPT', '受理', TRUE, '2026-06-03 11:30:00+08'::timestamptz, '2026-06-12 09:10:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '窗口已受理二级建造师增项注册申请，等待进入审查环节。')),
    ('AHZWFW-REAL10-003', 'rejected', 'MATERIAL_REVIEW', '材料预审', TRUE, '2026-06-05 10:15:00+08'::timestamptz, '2026-06-12 14:00:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '材料预审未通过，申请表信息需补正后重新提交。')),
    ('AHZWFW-REAL10-004', 'in_review', 'REVIEW', '审查', TRUE, '2026-06-06 09:20:00+08'::timestamptz, '2026-06-13 10:40:00+08'::timestamptz, jsonb_build_object('step', '审查', 'message', '二级建造师重新注册申请正在审查。')),
    ('AHZWFW-REAL10-005', 'completed', 'FINISH', '办结', TRUE, '2026-06-06 16:10:00+08'::timestamptz, '2026-06-06 16:10:00+08'::timestamptz, jsonb_build_object('step', '办结', 'message', '二级建造师延续注册已办结，结果已反馈。')),
    ('AHZWFW-REAL10-006', 'pre_accept', 'PRE_ACCEPT', '网上预审', TRUE, '2026-06-08 09:00:00+08'::timestamptz, '2026-06-13 15:30:00+08'::timestamptz, jsonb_build_object('step', '网上预审', 'message', '二级造价工程师注销注册正在网上预审。')),
    ('AHZWFW-REAL10-007', 'rejected', 'MATERIAL_REVIEW', '材料预审', TRUE, '2026-06-09 10:30:00+08'::timestamptz, '2026-06-13 16:05:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '承诺函未完成在线确认，已退回补正。')),
    ('AHZWFW-REAL10-008', 'in_decision', 'DECISION', '决定', TRUE, '2026-06-11 09:20:00+08'::timestamptz, '2026-06-14 11:50:00+08'::timestamptz, jsonb_build_object('step', '决定', 'message', '二级造价工程师初始注册已通过审查，等待作出核准决定。')),
    ('AHZWFW-REAL10-009', 'rejected', 'MATERIAL_REVIEW', '材料预审', TRUE, '2026-06-11 14:25:00+08'::timestamptz, '2026-06-14 15:40:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '延续注册申请表不符合受理标准，已退回补正。')),
    ('AHZWFW-REAL10-010', 'accepted', 'ACCEPT', '受理', TRUE, '2026-06-12 09:35:00+08'::timestamptz, '2026-06-14 17:10:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '二级结构工程师初始注册申请已受理。'))
)
INSERT INTO declaration_qa.business_application_statuses (
    apply_id,
    status,
    current_node,
    node_name,
    is_current,
    payload,
    created_at,
    updated_at
)
SELECT
    a.id,
    s.status,
    s.current_node,
    s.node_name,
    s.is_current,
    s.payload,
    s.created_at,
    s.updated_at
FROM seed_status s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid;

WITH seed_reject (
    apply_uid,
    material_code,
    material_name,
    reject_code,
    reject_reason,
    audit_node,
    is_current,
    created_at,
    updated_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-REAL10-003',
        'CONSTRUCTOR_INITIAL_APPLICATION_FORM',
        '二级建造师初始注册申请表',
        'FORM_CONTENT_INCONSISTENT',
        '申请表中注册专业和聘用单位信息填写不一致。请按示例样表核对后重新提交，表格内容需真实无误。',
        '材料预审',
        TRUE,
        '2026-06-12 13:30:00+08'::timestamptz,
        '2026-06-12 14:00:00+08'::timestamptz,
        jsonb_build_object('source_file', 'second_level_constructor_initial_registration.md', 'accept_standard', '表格内容填写真实无误。')
    ),
    (
        'AHZWFW-REAL10-007',
        'COMMITMENT_LETTER',
        '承诺函',
        'COMMITMENT_NOT_CONFIRMED',
        '承诺函未由企业或个人作为责任主体完成在线勾选确认。请重新进入申报系统完成承诺后提交。',
        '材料预审',
        TRUE,
        '2026-06-13 15:30:00+08'::timestamptz,
        '2026-06-13 16:05:00+08'::timestamptz,
        jsonb_build_object('source_file', 'second_level_cost_engineer_change_registration.md', 'accept_standard', '本承诺函需由企业或个人作为责任主体作出承诺，申报时通过在线勾选完成提交。')
    ),
    (
        'AHZWFW-REAL10-009',
        'COST_RENEWAL_APPLICATION_FORM',
        '二级造价工程师延续注册申请表',
        'CONTINUING_EDUCATION_NOT_CONFIRMED',
        '延续注册申请表未体现注册有效期届满30日内通过注册专业继续教育的相关信息，请补充后重新提交。',
        '材料预审',
        TRUE,
        '2026-06-14 15:05:00+08'::timestamptz,
        '2026-06-14 15:40:00+08'::timestamptz,
        jsonb_build_object('source_file', 'second_level_cost_engineer_renewal_registration.md', 'accept_standard', '在注册有效期届满30日内，通过注册专业继续教育后申请延续注册。')
    )
)
INSERT INTO declaration_qa.business_reject_infos (
    apply_id,
    material_code,
    material_name,
    reject_code,
    reject_reason,
    audit_node,
    is_current,
    payload,
    created_at,
    updated_at
)
SELECT
    a.id,
    r.material_code,
    r.material_name,
    r.reject_code,
    r.reject_reason,
    r.audit_node,
    r.is_current,
    r.payload,
    r.created_at,
    r.updated_at
FROM seed_reject r
JOIN declaration_qa.business_applications a ON a.apply_uid = r.apply_uid;

COMMIT;


-- ============================================================================
-- 来源：005_declaration_qa_seed_business_mock_fire_rejected.sql
-- ============================================================================

-- 用于测试消防流程路由的虚拟退回业务工单。
--
-- 此种子数据特意限定为 apply_uid LIKE 'AHZWFW-MOCK-FIRE-%'，因此可重复运行，
-- 且不会影响真实业务工单。

BEGIN;

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-MOCK-FIRE-%'
)
DELETE FROM declaration_qa.business_reject_infos
WHERE apply_id IN (SELECT id FROM target_apps);

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-MOCK-FIRE-%'
)
DELETE FROM declaration_qa.business_application_statuses
WHERE apply_id IN (SELECT id FROM target_apps);

WITH seed (
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-MOCK-FIRE-001',
        1,
        'grassland_fire_blasting_approval',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-17 09:20:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'title', '防火办事流程',
            'item_name', '防火办事流程',
            'official_item_name', '草原防火期内在草原上进行爆破、勘察和施工等活动审批',
            'item_code', '11340000002985990J200016407900001',
            'source_file', 'item_30.md',
            'materials', jsonb_build_array('身份证件', '单位组织机构代码证', '申请表', '市县林业主管部门审查审核意见', '民用爆炸物品购买许可证', '爆破作业单位许可证'),
            'legal_limit', '20个工作日',
            'promise_limit', '10个工作日',
            'handle_form', '网上办理,移动端办理,窗口办理,快递申请',
            'consult_phone', '0551-62999817',
            'complaint_phone', '0551-62634463',
            'is_mock', true
        )
    )
)
INSERT INTO declaration_qa.business_applications (
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    payload,
    deleted_at,
    created_at,
    updated_at
)
SELECT
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at::timestamptz,
    completed_at::timestamptz,
    payload,
    NULL::timestamptz,
    submitted_at::timestamptz,
    '2026-06-17 10:30:00+08'::timestamptz
FROM seed
ON CONFLICT (apply_uid) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    biz_type = EXCLUDED.biz_type,
    region_code = EXCLUDED.region_code,
    status = EXCLUDED.status,
    current_node = EXCLUDED.current_node,
    node_name = EXCLUDED.node_name,
    submitted_at = EXCLUDED.submitted_at,
    completed_at = EXCLUDED.completed_at,
    payload = EXCLUDED.payload,
    deleted_at = NULL,
    updated_at = EXCLUDED.updated_at;

WITH seed_status (
    apply_uid,
    status,
    current_node,
    node_name,
    is_current,
    created_at,
    updated_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-MOCK-FIRE-001',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        TRUE,
        '2026-06-17 10:00:00+08'::timestamptz,
        '2026-06-17 10:30:00+08'::timestamptz,
        jsonb_build_object('step', '材料预审', 'message', '防火办事流程申请材料预审未通过，已退回补正。')
    )
)
INSERT INTO declaration_qa.business_application_statuses (
    apply_id,
    status,
    current_node,
    node_name,
    is_current,
    payload,
    created_at,
    updated_at
)
SELECT
    a.id,
    s.status,
    s.current_node,
    s.node_name,
    s.is_current,
    s.payload,
    s.created_at,
    s.updated_at
FROM seed_status s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid;

WITH seed_reject (
    apply_uid,
    material_code,
    material_name,
    reject_code,
    reject_reason,
    audit_node,
    is_current,
    created_at,
    updated_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-MOCK-FIRE-001',
        'FIRE_PROCESS_APPLICATION_FORM',
        '申请表',
        'FIRE_MEASURE_DESCRIPTION_INCOMPLETE',
        '申请表中的活动地点、作业时间和防火措施说明不完整，未能证明已采取必要防火措施。请补充具体作业地点、时间安排、防火责任人和现场防火措施后重新提交。',
        '材料预审',
        TRUE,
        '2026-06-17 10:10:00+08'::timestamptz,
        '2026-06-17 10:30:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'item_30.md',
            'official_item_name', '草原防火期内在草原上进行爆破、勘察和施工等活动审批',
            'accept_standard', '申请表应填写齐全，单位申请要加盖公章，内容真实详细准确。',
            'is_mock', true
        )
    )
)
INSERT INTO declaration_qa.business_reject_infos (
    apply_id,
    material_code,
    material_name,
    reject_code,
    reject_reason,
    audit_node,
    is_current,
    payload,
    created_at,
    updated_at
)
SELECT
    a.id,
    r.material_code,
    r.material_name,
    r.reject_code,
    r.reject_reason,
    r.audit_node,
    r.is_current,
    r.payload,
    r.created_at,
    r.updated_at
FROM seed_reject r
JOIN declaration_qa.business_applications a ON a.apply_uid = r.apply_uid;

COMMIT;


-- ============================================================================
-- 来源：006_declaration_qa_fix_business_item_code_manifest_id.sql
-- ============================================================================

-- 修复 payload.item_code 误存整理后文档清单标识的业务工单载荷，
-- 例如 *_official_ahzwfw。
--
-- 此迁移完成后：
--   payload.item_code           = chunks.metadata.item_code 使用的政府官方事项编码
--   payload.source_manifest_id  = 整理后文档的清单标识
--   payload.source_file         = 能够解析时对应的 Markdown 来源文件名
--
-- 此脚本具备幂等性，并且只修改当前 item_code 为清单样式标识的记录。

BEGIN;

WITH chunk_scope AS (
    SELECT
        c.metadata->>'item_name' AS item_name,
        c.metadata->>'item_code' AS official_item_code,
        MIN(d.filename) AS source_file
    FROM declaration_qa.chunks c
    JOIN declaration_qa.documents d ON d.id = c.document_id
    WHERE COALESCE(c.metadata->>'item_name', '') <> ''
      AND COALESCE(c.metadata->>'item_code', '') <> ''
      AND COALESCE(c.metadata->>'item_code', '') !~ '^[a-z0-9_]+_official_ahzwfw$'
    GROUP BY c.metadata->>'item_name', c.metadata->>'item_code'
),
chunk_mappings AS (
    SELECT
        a.id AS application_id,
        c.official_item_code,
        c.source_file,
        1 AS priority
    FROM declaration_qa.business_applications a
    JOIN chunk_scope c
      ON c.item_name = COALESCE(
          NULLIF(a.payload->>'official_item_name', ''),
          NULLIF(a.payload->>'item_name', ''),
          NULLIF(a.payload->>'title', '')
      )
    WHERE COALESCE(a.payload->>'item_code', '') ~ '^[a-z0-9_]+_official_ahzwfw$'
      AND a.deleted_at IS NULL
),
known_manifest_mappings (
    source_manifest_id,
    official_item_code,
    source_file
) AS (
    VALUES
    ('foreign_related_adoption_registration_official_ahzwfw', '11340000002986424J200071100200001', 'foreign_related_adoption_registration.md'),
    ('overseas_chinese_return_settlement_approval_official_ahzwfw', '11340000002986483M200014400100001', 'overseas_chinese_return_settlement_approval.md'),
    ('foreigners_group_religious_activity_temporary_site_approval_official_ahzwfw', '11340000MB0U046663200014100200001', 'foreigners_group_religious_activity_temporary_site_approval.md'),
    ('science_technology_archive_query_official_ahzwfw', '11340000002986088E234200603400001', 'science_technology_archive_query.md'),
    ('mobile_personnel_archive_material_collection_official_ahzwfw', '11340000002986547K200201420400201', 'mobile_personnel_archive_material_collection.md'),
    ('pine_wood_nematode_infected_timber_processing_change_approval_official_ahzwfw', '11340000002985990J200016413700001', 'pine_wood_nematode_infected_timber_processing_change_approval.md'),
    ('high_risk_sports_event_permit_official_ahzwfw', '11340000002985974W234013300500001', 'high_risk_sports_event_permit.md'),
    ('second_level_constructor_initial_registration_official_ahzwfw', '11340000MB0T3248X7200011705500205', 'second_level_constructor_initial_registration.md'),
    ('foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', '11340000002986424J200071100100002', 'foreign_hmt_overseas_chinese_marriage_registration.md'),
    ('disabled_veterans_centralized_support_determination_official_ahzwfw', '11340000MB1527779M200072400100001', 'disabled_veterans_centralized_support_determination.md')
),
manifest_mappings AS (
    SELECT
        a.id AS application_id,
        m.official_item_code,
        m.source_file,
        2 AS priority
    FROM declaration_qa.business_applications a
    JOIN known_manifest_mappings m
      ON m.source_manifest_id = a.payload->>'item_code'
    WHERE a.deleted_at IS NULL
),
resolved_mappings AS (
    SELECT DISTINCT ON (application_id)
        application_id,
        official_item_code,
        source_file
    FROM (
        SELECT * FROM chunk_mappings
        UNION ALL
        SELECT * FROM manifest_mappings
    ) mappings
    ORDER BY application_id, priority
)
UPDATE declaration_qa.business_applications a
SET
    payload = jsonb_set(
        jsonb_set(
            jsonb_set(
                a.payload,
                '{source_manifest_id}',
                to_jsonb(COALESCE(NULLIF(a.payload->>'source_manifest_id', ''), a.payload->>'item_code')),
                true
            ),
            '{item_code}',
            to_jsonb(r.official_item_code),
            true
        ),
        '{source_file}',
        to_jsonb(COALESCE(NULLIF(r.source_file, ''), a.payload->>'source_file')),
        true
    ),
    updated_at = CURRENT_TIMESTAMP
FROM resolved_mappings r
WHERE a.id = r.application_id
  AND COALESCE(a.payload->>'item_code', '') ~ '^[a-z0-9_]+_official_ahzwfw$';

COMMIT;


-- ============================================================================
-- 来源：008_declaration_qa_agentic_ticket_history.sql
-- ============================================================================

-- Agent 工单工具要求按事项查询申请，并为每条申请提供完整、稳定的状态时间线。
-- 此迁移具备幂等性。

BEGIN;

ALTER TABLE declaration_qa.business_applications
    ADD COLUMN IF NOT EXISTS item_code VARCHAR(160) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS item_name VARCHAR(255) NOT NULL DEFAULT '';

UPDATE declaration_qa.business_applications
SET
    item_code = COALESCE(NULLIF(item_code, ''), payload->>'item_code', ''),
    item_name = COALESCE(
        NULLIF(item_name, ''),
        payload->>'item_name',
        payload->>'official_item_name',
        payload->>'title',
        ''
    )
WHERE item_code = '' OR item_name = '';

CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_applications_user_item
    ON declaration_qa.business_applications (user_id, item_code, updated_at DESC)
    WHERE deleted_at IS NULL;

ALTER TABLE declaration_qa.business_application_statuses
    ADD COLUMN IF NOT EXISTS sequence_no INTEGER,
    ADD COLUMN IF NOT EXISTS from_status VARCHAR(40) NULL,
    ADD COLUMN IF NOT EXISTS reason_code VARCHAR(80) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS reason TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS material_code VARCHAR(80) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS material_name VARCHAR(160) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS source_event_id VARCHAR(160) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMPTZ;

WITH ordered AS (
    SELECT
        id,
        ROW_NUMBER() OVER (PARTITION BY apply_id ORDER BY created_at, id) AS generated_sequence,
        LAG(status) OVER (PARTITION BY apply_id ORDER BY created_at, id) AS generated_from_status
    FROM declaration_qa.business_application_statuses
)
UPDATE declaration_qa.business_application_statuses s
SET
    sequence_no = COALESCE(s.sequence_no, ordered.generated_sequence),
    from_status = COALESCE(s.from_status, ordered.generated_from_status),
    occurred_at = COALESCE(s.occurred_at, s.created_at)
FROM ordered
WHERE ordered.id = s.id;

UPDATE declaration_qa.business_application_statuses s
SET
    reason_code = COALESCE(NULLIF(s.reason_code, ''), r.reject_code, ''),
    reason = COALESCE(NULLIF(s.reason, ''), r.reject_reason, ''),
    material_code = COALESCE(NULLIF(s.material_code, ''), r.material_code, ''),
    material_name = COALESCE(NULLIF(s.material_name, ''), r.material_name, '')
FROM declaration_qa.business_reject_infos r
WHERE r.apply_id = s.apply_id
  AND r.is_current = TRUE
  AND s.is_current = TRUE;

ALTER TABLE declaration_qa.business_application_statuses
    ALTER COLUMN sequence_no SET NOT NULL,
    ALTER COLUMN sequence_no SET DEFAULT 1,
    ALTER COLUMN occurred_at SET NOT NULL,
    ALTER COLUMN occurred_at SET DEFAULT CURRENT_TIMESTAMP;

CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_business_status_sequence
    ON declaration_qa.business_application_statuses (apply_id, sequence_no);
DROP INDEX IF EXISTS declaration_qa.ux_declaration_qa_business_status_source_event;
CREATE UNIQUE INDEX ux_declaration_qa_business_status_source_event
    ON declaration_qa.business_application_statuses (apply_id, source_event_id)
    WHERE source_event_id <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_status_timeline
    ON declaration_qa.business_application_statuses (apply_id, occurred_at, sequence_no);
CREATE OR REPLACE FUNCTION declaration_qa.prepare_business_application_status_event()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    previous_sequence INTEGER;
    previous_status VARCHAR(40);
BEGIN
    PERFORM pg_advisory_xact_lock(NEW.apply_id);

    IF NEW.source_event_id <> ''
       AND EXISTS (
           SELECT 1
           FROM declaration_qa.business_application_statuses
           WHERE apply_id = NEW.apply_id
             AND source_event_id = NEW.source_event_id
       ) THEN
        RETURN NULL;
    END IF;

    SELECT sequence_no, status
    INTO previous_sequence, previous_status
    FROM declaration_qa.business_application_statuses
    WHERE apply_id = NEW.apply_id
    ORDER BY sequence_no DESC, occurred_at DESC, id DESC
    LIMIT 1;

    IF NEW.sequence_no IS NULL
       OR NEW.sequence_no <= COALESCE(previous_sequence, 0) THEN
        NEW.sequence_no := COALESCE(previous_sequence, 0) + 1;
    END IF;
    NEW.from_status := COALESCE(NEW.from_status, previous_status);
    NEW.occurred_at := COALESCE(NEW.occurred_at, CURRENT_TIMESTAMP);

    IF NEW.is_current THEN
        UPDATE declaration_qa.business_application_statuses
        SET is_current = FALSE,
            updated_at = CURRENT_TIMESTAMP
        WHERE apply_id = NEW.apply_id
          AND is_current = TRUE;
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_declaration_qa_prepare_business_status
    ON declaration_qa.business_application_statuses;
CREATE TRIGGER trg_declaration_qa_prepare_business_status
BEFORE INSERT ON declaration_qa.business_application_statuses
FOR EACH ROW
EXECUTE FUNCTION declaration_qa.prepare_business_application_status_event();

CREATE OR REPLACE FUNCTION declaration_qa.sync_business_application_current_status()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.is_current THEN
        UPDATE declaration_qa.business_applications
        SET status = NEW.status,
            current_node = NEW.current_node,
            node_name = NEW.node_name,
            updated_at = GREATEST(updated_at, NEW.occurred_at)
        WHERE id = NEW.apply_id;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_declaration_qa_sync_business_status
    ON declaration_qa.business_application_statuses;
CREATE TRIGGER trg_declaration_qa_sync_business_status
AFTER INSERT ON declaration_qa.business_application_statuses
FOR EACH ROW
EXECUTE FUNCTION declaration_qa.sync_business_application_current_status();


ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_plan JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS agent_query_plan JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS tool_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS agent_iteration INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS agent_answer_provider VARCHAR(120) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS agent_selected_item_code VARCHAR(160) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS agent_selected_application_id BIGINT NULL;

CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_tool_results_gin
    ON declaration_qa.audit_logs USING GIN (tool_results);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_agent_query_plan_gin
    ON declaration_qa.audit_logs USING GIN (agent_query_plan);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_agent_item
    ON declaration_qa.audit_logs (agent_selected_item_code, created_at DESC);

COMMIT;


-- ============================================================================
-- 来源：009_declaration_qa_agent_message_history.sql
-- ============================================================================

-- 持久化可重放的 Agent 运行记录，同时不在普通聊天历史中暴露内部消息。
-- 此迁移具备幂等性，并保留所有现有会话消息。

BEGIN;

CREATE TABLE IF NOT EXISTS declaration_qa.agent_runs (
    id BIGSERIAL PRIMARY KEY,
    run_uid VARCHAR(80) NOT NULL UNIQUE,
    session_id BIGINT NOT NULL REFERENCES declaration_qa.sessions(id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    iteration_count INTEGER NOT NULL DEFAULT 0,
    trace_id VARCHAR(120) NOT NULL DEFAULT '',
    answer_provider VARCHAR(120) NOT NULL DEFAULT '',
    idempotency_key VARCHAR(160) NOT NULL DEFAULT '',
    error_code VARCHAR(120) NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    prompt_version VARCHAR(80) NOT NULL DEFAULT '',
    tool_registry_version VARCHAR(80) NOT NULL DEFAULT '',
    model_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_agent_runs_session
    ON declaration_qa.agent_runs (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_agent_runs_user
    ON declaration_qa.agent_runs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_agent_runs_trace
    ON declaration_qa.agent_runs (trace_id)
    WHERE trace_id <> '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_agent_runs_idempotency
    ON declaration_qa.agent_runs (session_id, idempotency_key)
    WHERE idempotency_key <> '';

ALTER TABLE declaration_qa.messages
    ADD COLUMN IF NOT EXISTS message_uid VARCHAR(80),
    ADD COLUMN IF NOT EXISTS agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS sequence_no INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS parent_message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS message_type VARCHAR(30) NOT NULL DEFAULT 'conversation',
    ADD COLUMN IF NOT EXISTS content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'completed',
    ADD COLUMN IF NOT EXISTS visible BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS model VARCHAR(160) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS provider VARCHAR(120) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS input_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS output_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS total_tokens INTEGER NULL,
    ADD COLUMN IF NOT EXISTS latency_ms BIGINT NULL,
    ADD COLUMN IF NOT EXISTS pii_level VARCHAR(30) NOT NULL DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS retention_until TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(160) NOT NULL DEFAULT '';

UPDATE declaration_qa.messages
SET message_uid = 'legacy_message_' || id::text
WHERE message_uid IS NULL OR message_uid = '';

ALTER TABLE declaration_qa.messages
    ALTER COLUMN message_uid SET NOT NULL,
    DROP CONSTRAINT IF EXISTS messages_role_check,
    DROP CONSTRAINT IF EXISTS messages_status_check;
ALTER TABLE declaration_qa.messages
    ADD CONSTRAINT messages_role_check CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    ADD CONSTRAINT messages_status_check CHECK (status IN ('pending', 'completed', 'failed', 'cancelled'));

CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_messages_uid
    ON declaration_qa.messages (message_uid);
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_messages_run_sequence
    ON declaration_qa.messages (agent_run_id, sequence_no)
    WHERE agent_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_run
    ON declaration_qa.messages (agent_run_id, sequence_no);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_messages_parent
    ON declaration_qa.messages (parent_message_id);

CREATE TABLE IF NOT EXISTS declaration_qa.tool_calls (
    id BIGSERIAL PRIMARY KEY,
    tool_call_uid VARCHAR(100) NOT NULL UNIQUE,
    agent_run_id BIGINT NOT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE CASCADE,
    assistant_message_id BIGINT NOT NULL REFERENCES declaration_qa.messages(id) ON DELETE CASCADE,
    result_message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL,
    external_call_id VARCHAR(160) NOT NULL DEFAULT '',
    sequence_no INTEGER NOT NULL,
    tool_name VARCHAR(160) NOT NULL,
    arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(80) NOT NULL DEFAULT 'pending',
    ok BOOLEAN NOT NULL DEFAULT FALSE,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT NOT NULL DEFAULT '',
    latency_ms BIGINT NULL CHECK (latency_ms IS NULL OR latency_ms >= 0),
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agent_run_id, sequence_no)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_tool_calls_run
    ON declaration_qa.tool_calls (agent_run_id, sequence_no);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_tool_calls_external
    ON declaration_qa.tool_calls (agent_run_id, external_call_id);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_tool_calls_name
    ON declaration_qa.tool_calls (tool_name, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.message_parts (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES declaration_qa.messages(id) ON DELETE CASCADE,
    part_index INTEGER NOT NULL,
    part_type VARCHAR(30) NOT NULL,
    text_content TEXT NOT NULL DEFAULT '',
    asset_uri TEXT NOT NULL DEFAULT '',
    mime_type VARCHAR(160) NOT NULL DEFAULT '',
    file_name VARCHAR(500) NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (message_id, part_index)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_parts_message
    ON declaration_qa.message_parts (message_id, part_index);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_parts_type
    ON declaration_qa.message_parts (part_type);

CREATE TABLE IF NOT EXISTS declaration_qa.message_sources (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES declaration_qa.messages(id) ON DELETE CASCADE,
    citation_no INTEGER NOT NULL,
    chunk_uid VARCHAR(160) NOT NULL DEFAULT '',
    document_uid VARCHAR(160) NOT NULL DEFAULT '',
    source_asset_id VARCHAR(200) NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    page_no INTEGER NULL,
    relevance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (message_id, citation_no)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_sources_message
    ON declaration_qa.message_sources (message_id, citation_no);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_sources_chunk
    ON declaration_qa.message_sources (chunk_uid)
    WHERE chunk_uid <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_message_sources_asset
    ON declaration_qa.message_sources (source_asset_id)
    WHERE source_asset_id <> '';

ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE SET NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_audit_logs_agent_run
    ON declaration_qa.audit_logs (agent_run_id)
    WHERE agent_run_id IS NOT NULL;

ALTER TABLE declaration_qa.feedback
    ADD COLUMN IF NOT EXISTS agent_run_id BIGINT NULL REFERENCES declaration_qa.agent_runs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS message_id BIGINT NULL REFERENCES declaration_qa.messages(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_agent_run
    ON declaration_qa.feedback (agent_run_id)
    WHERE agent_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_feedback_message
    ON declaration_qa.feedback (message_id)
    WHERE message_id IS NOT NULL;

COMMENT ON TABLE declaration_qa.agent_runs IS '一次用户请求对应的 Agent 执行记录';
COMMENT ON TABLE declaration_qa.tool_calls IS 'Agent 工具调用及其执行生命周期';
COMMENT ON TABLE declaration_qa.message_parts IS '消息的文本、图片、文件和 OCR 多模态组成部分';
COMMENT ON TABLE declaration_qa.message_sources IS '最终回答消息与知识证据的引用关系';
COMMENT ON COLUMN declaration_qa.messages.visible IS '是否出现在普通会话历史中';
COMMENT ON COLUMN declaration_qa.messages.content_json IS '工具调用等结构化消息内容';

COMMIT;


-- ============================================================================
-- 来源：010_declaration_qa_seed_business_doc40_mock.sql
-- ============================================================================

-- 为当前已发布的四份 AHZWFW 文档生成模拟工单。
--
-- 此种子数据面向演示用户 user_id=1，并具备幂等性。重复运行时只会替换
-- apply_uid 以 AHZWFW-DOC40-MOCK- 开头的申请。

BEGIN;

DELETE FROM declaration_qa.business_applications
WHERE apply_uid LIKE 'AHZWFW-DOC40-MOCK-%';

WITH seed (
    apply_uid,
    user_id,
    item_code,
    item_name,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    updated_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-DOC40-MOCK-OCS-2025', 1,
        '11340000002986483M200014400100001', '华侨回国定居审批',
        'overseas_chinese_return_settlement_approval', '340000',
        'completed', 'FINISH', '办结',
        '2025-03-02 09:10:00+08'::timestamptz,
        '2025-03-20 16:30:00+08'::timestamptz,
        '2025-03-20 16:30:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'overseas_chinese_return_settlement_approval.md',
            'materials', jsonb_build_array('华侨回国定居申请表', '自愿放弃国外居留资格声明书', '中华人民共和国护照', '国外居留证明', '户口注销书面说明', '稳定生活保障证明', '房产证', '出入境记录'),
            'result', '准予华侨回国定居'
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-OCS-2026', 1,
        '11340000002986483M200014400100001', '华侨回国定居审批',
        'overseas_chinese_return_settlement_approval', '340000',
        'returned_for_correction', 'MATERIAL_REVIEW', '材料预审',
        '2026-07-01 10:05:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-10 14:20:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'overseas_chinese_return_settlement_approval.md',
            'materials', jsonb_build_array('华侨回国定居申请表', '自愿放弃国外居留资格声明书', '中华人民共和国护照', '国外居留证明', '户口注销书面说明', '稳定生活保障证明', '房产证', '出入境记录')
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-WATER-2026', 1,
        '11340000002985923L200011900100001', '取水许可',
        'water_withdrawal_permit', '340000',
        'in_review', 'TECHNICAL_REVIEW', '技术审查',
        '2026-07-03 09:25:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-11 11:40:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'item_28.md',
            'materials', jsonb_build_array('建设项目水资源论证报告书（表）', '水权转让可行性研究报告', '备案材料', '营业执照或身份证', '取水许可申请书', '利害关系说明')
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-ADOPTION-2025', 1,
        '11340000002986424J200071100200001', '涉外收养登记',
        'foreign_related_adoption_registration', '340000',
        'completed', 'FINISH', '办结',
        '2025-08-12 09:00:00+08'::timestamptz,
        '2025-08-18 15:10:00+08'::timestamptz,
        '2025-08-18 15:10:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'foreign_related_adoption_registration.md',
            'materials', jsonb_build_array('收养人护照', '来华收养子女通知书', '收养人照片', '送养人身份证', '被收养人照片', '授权委托书'),
            'result', '涉外收养登记完成'
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-ADOPTION-2026', 1,
        '11340000002986424J200071100200001', '涉外收养登记',
        'foreign_related_adoption_registration', '340000',
        'returned_for_correction', 'MATERIAL_REVIEW', '材料预审',
        '2026-07-05 14:15:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-09 16:45:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'foreign_related_adoption_registration.md',
            'materials', jsonb_build_array('收养人护照', '来华收养子女通知书', '收养人照片', '送养人身份证', '被收养人照片', '授权委托书')
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-MARRIAGE-2026', 1,
        '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记',
        'foreign_hmt_overseas_chinese_marriage_registration', '340000',
        'accepted', 'ACCEPT', '受理',
        '2026-07-08 10:30:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-08 11:05:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'foreign_hmt_overseas_chinese_marriage_registration.md',
            'materials', jsonb_build_array('内地居民身份证', '港澳台居民身份证', '港澳台居民通行证', '护照', '无血亲关系证明', '单身证明', '合照')
        )
    )
)
INSERT INTO declaration_qa.business_applications (
    apply_uid, user_id, item_code, item_name, biz_type, region_code,
    status, current_node, node_name, submitted_at, completed_at,
    payload, deleted_at, created_at, updated_at
)
SELECT
    apply_uid, user_id, item_code, item_name, biz_type, region_code,
    status, current_node, node_name, submitted_at, completed_at,
    payload || jsonb_build_object('item_code', item_code, 'item_name', item_name),
    NULL::timestamptz, submitted_at, updated_at
FROM seed;

WITH seed_status (
    apply_uid, sequence_no, from_status, status, current_node, node_name,
    reason_code, reason, material_code, material_name, is_current,
    occurred_at, source_event_id
) AS (
    VALUES
    ('AHZWFW-DOC40-MOCK-OCS-2025', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2025-03-02 09:10:00+08'::timestamptz, 'DOC40-OCS-2025-01'),
    ('AHZWFW-DOC40-MOCK-OCS-2025', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '', '', '', FALSE, '2025-03-04 10:20:00+08'::timestamptz, 'DOC40-OCS-2025-02'),
    ('AHZWFW-DOC40-MOCK-OCS-2025', 3, 'accepted', 'in_review', 'REVIEW', '审查', '', '', '', '', FALSE, '2025-03-10 09:30:00+08'::timestamptz, 'DOC40-OCS-2025-03'),
    ('AHZWFW-DOC40-MOCK-OCS-2025', 4, 'in_review', 'completed', 'FINISH', '办结', '', '审查通过，准予华侨回国定居。', '', '', TRUE, '2025-03-20 16:30:00+08'::timestamptz, 'DOC40-OCS-2025-04'),

    ('AHZWFW-DOC40-MOCK-OCS-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-01 10:05:00+08'::timestamptz, 'DOC40-OCS-2026-01'),
    ('AHZWFW-DOC40-MOCK-OCS-2026', 2, 'submitted', 'pre_accept', 'PRE_ACCEPT', '网上预审', '', '', '', '', FALSE, '2026-07-02 09:15:00+08'::timestamptz, 'DOC40-OCS-2026-02'),
    ('AHZWFW-DOC40-MOCK-OCS-2026', 3, 'pre_accept', 'returned_for_correction', 'MATERIAL_REVIEW', '材料预审', 'MISSING_PROPERTY_AND_TRAVEL_RECORD', '未提交房产证，且出入境记录缺少最近五年的完整记录，请补充后重新提交。', 'PROPERTY_AND_TRAVEL_RECORD', '房产证、出入境记录', TRUE, '2026-07-10 14:20:00+08'::timestamptz, 'DOC40-OCS-2026-03'),

    ('AHZWFW-DOC40-MOCK-WATER-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-03 09:25:00+08'::timestamptz, 'DOC40-WATER-2026-01'),
    ('AHZWFW-DOC40-MOCK-WATER-2026', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '', '', '', FALSE, '2026-07-04 14:00:00+08'::timestamptz, 'DOC40-WATER-2026-02'),
    ('AHZWFW-DOC40-MOCK-WATER-2026', 3, 'accepted', 'in_review', 'TECHNICAL_REVIEW', '技术审查', '', '正在审查建设项目水资源论证报告书及取水合理性。', '', '', TRUE, '2026-07-11 11:40:00+08'::timestamptz, 'DOC40-WATER-2026-03'),

    ('AHZWFW-DOC40-MOCK-ADOPTION-2025', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2025-08-12 09:00:00+08'::timestamptz, 'DOC40-ADOPTION-2025-01'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2025', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '', '', '', FALSE, '2025-08-13 10:00:00+08'::timestamptz, 'DOC40-ADOPTION-2025-02'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2025', 3, 'accepted', 'completed', 'FINISH', '办结', '', '材料审查通过，涉外收养登记已完成。', '', '', TRUE, '2025-08-18 15:10:00+08'::timestamptz, 'DOC40-ADOPTION-2025-03'),

    ('AHZWFW-DOC40-MOCK-ADOPTION-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-05 14:15:00+08'::timestamptz, 'DOC40-ADOPTION-2026-01'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2026', 2, 'submitted', 'pre_accept', 'PRE_ACCEPT', '网上预审', '', '', '', '', FALSE, '2026-07-06 09:40:00+08'::timestamptz, 'DOC40-ADOPTION-2026-02'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2026', 3, 'pre_accept', 'returned_for_correction', 'MATERIAL_REVIEW', '材料预审', 'AUTHORIZATION_UNSIGNED', '授权委托书缺少委托人签字，被收养人照片也未按要求提交，请补正后重新申报。', 'AUTHORIZATION_AND_PHOTO', '授权委托书、被收养人照片', TRUE, '2026-07-09 16:45:00+08'::timestamptz, 'DOC40-ADOPTION-2026-03'),

    ('AHZWFW-DOC40-MOCK-MARRIAGE-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-08 10:30:00+08'::timestamptz, 'DOC40-MARRIAGE-2026-01'),
    ('AHZWFW-DOC40-MOCK-MARRIAGE-2026', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '申请材料已核验，等待登记办理。', '', '', TRUE, '2026-07-08 11:05:00+08'::timestamptz, 'DOC40-MARRIAGE-2026-02')
)
INSERT INTO declaration_qa.business_application_statuses (
    apply_id, sequence_no, from_status, status, current_node, node_name,
    reason_code, reason, material_code, material_name, source_event_id,
    is_current, payload, occurred_at, created_at, updated_at
)
SELECT
    a.id, s.sequence_no, s.from_status, s.status, s.current_node, s.node_name,
    s.reason_code, s.reason, s.material_code, s.material_name, s.source_event_id,
    s.is_current, jsonb_build_object('source', 'doc40_mock_seed'),
    s.occurred_at, s.occurred_at, s.occurred_at
FROM seed_status s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid
ORDER BY s.apply_uid, s.sequence_no;

WITH seed_reject (
    apply_uid, material_code, material_name, reject_code, reject_reason,
    audit_node, created_at
) AS (
    VALUES
    (
        'AHZWFW-DOC40-MOCK-OCS-2026',
        'PROPERTY_AND_TRAVEL_RECORD', '房产证、出入境记录',
        'MISSING_PROPERTY_AND_TRAVEL_RECORD',
        '未提交房产证，且出入境记录缺少最近五年的完整记录，请补充后重新提交。',
        '材料预审', '2026-07-10 14:20:00+08'::timestamptz
    ),
    (
        'AHZWFW-DOC40-MOCK-ADOPTION-2026',
        'AUTHORIZATION_AND_PHOTO', '授权委托书、被收养人照片',
        'AUTHORIZATION_UNSIGNED',
        '授权委托书缺少委托人签字，被收养人照片也未按要求提交，请补正后重新申报。',
        '材料预审', '2026-07-09 16:45:00+08'::timestamptz
    )
)
INSERT INTO declaration_qa.business_reject_infos (
    apply_id, material_code, material_name, reject_code, reject_reason,
    audit_node, is_current, payload, created_at, updated_at
)
SELECT
    a.id, r.material_code, r.material_name, r.reject_code, r.reject_reason,
    r.audit_node, TRUE, jsonb_build_object('source', 'doc40_mock_seed'),
    r.created_at, r.created_at
FROM seed_reject r
JOIN declaration_qa.business_applications a ON a.apply_uid = r.apply_uid;

COMMIT;

-- ============================================================================
-- 来源：005_business_query_tools.sql
-- ============================================================================

-- 预约办理和只读服务查询工具使用的业务数据。

BEGIN;

CREATE TABLE IF NOT EXISTS declaration_qa.business_appointment_slots (
    id BIGSERIAL PRIMARY KEY,
    slot_uid VARCHAR(120) NOT NULL UNIQUE,
    item_code VARCHAR(160) NOT NULL,
    item_name VARCHAR(255) NOT NULL DEFAULT '',
    region_code VARCHAR(20) NOT NULL DEFAULT '340000',
    hall_code VARCHAR(80) NOT NULL,
    hall_name VARCHAR(255) NOT NULL,
    hall_address TEXT NOT NULL DEFAULT '',
    service_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    capacity INTEGER NOT NULL DEFAULT 0 CHECK (capacity >= 0),
    reserved_count INTEGER NOT NULL DEFAULT 0 CHECK (reserved_count >= 0),
    status VARCHAR(40) NOT NULL DEFAULT 'available',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_appointment_slots_lookup
    ON declaration_qa.business_appointment_slots (item_code, region_code, service_date, hall_code);

CREATE TABLE IF NOT EXISTS declaration_qa.business_appointments (
    id BIGSERIAL PRIMARY KEY,
    appointment_uid VARCHAR(120) NOT NULL UNIQUE,
    slot_id BIGINT NULL REFERENCES declaration_qa.business_appointment_slots(id) ON DELETE RESTRICT,
    idempotency_key VARCHAR(120) NOT NULL DEFAULT '',
    user_id BIGINT NOT NULL,
    item_code VARCHAR(160) NOT NULL,
    item_name VARCHAR(255) NOT NULL DEFAULT '',
    region_code VARCHAR(20) NOT NULL DEFAULT '340000',
    hall_code VARCHAR(80) NOT NULL,
    hall_name VARCHAR(255) NOT NULL,
    hall_address TEXT NOT NULL DEFAULT '',
    service_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    status VARCHAR(40) NOT NULL DEFAULT 'booked',
    queue_number VARCHAR(40) NOT NULL DEFAULT '',
    people_ahead INTEGER NOT NULL DEFAULT 0 CHECK (people_ahead >= 0),
    estimated_wait_minutes INTEGER NOT NULL DEFAULT 0 CHECK (estimated_wait_minutes >= 0),
    counter_name VARCHAR(120) NOT NULL DEFAULT '',
    checked_in_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_appointments_user
    ON declaration_qa.business_appointments (user_id, service_date DESC, updated_at DESC);
ALTER TABLE declaration_qa.business_appointment_slots
    ADD COLUMN IF NOT EXISTS item_name VARCHAR(255) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.business_appointments
    ADD COLUMN IF NOT EXISTS slot_id BIGINT NULL REFERENCES declaration_qa.business_appointment_slots(id) ON DELETE RESTRICT;
ALTER TABLE declaration_qa.business_appointments
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(120) NOT NULL DEFAULT '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_appointments_idempotency
    ON declaration_qa.business_appointments (user_id, idempotency_key)
    WHERE idempotency_key <> '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_appointments_active_slot
    ON declaration_qa.business_appointments (user_id, slot_id)
    WHERE slot_id IS NOT NULL AND status IN ('booked', 'checked_in', 'waiting', 'called');

CREATE TABLE IF NOT EXISTS declaration_qa.business_payment_orders (
    id BIGSERIAL PRIMARY KEY,
    order_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    order_no VARCHAR(120) NOT NULL UNIQUE,
    fee_item VARCHAR(255) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL DEFAULT 0 CHECK (amount >= 0),
    currency VARCHAR(12) NOT NULL DEFAULT 'CNY',
    status VARCHAR(40) NOT NULL DEFAULT 'pending',
    payment_channel VARCHAR(80) NOT NULL DEFAULT '',
    payable_at TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NULL,
    paid_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_payment_orders_apply
    ON declaration_qa.business_payment_orders (apply_id, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_payment_receipts (
    id BIGSERIAL PRIMARY KEY,
    receipt_uid VARCHAR(120) NOT NULL UNIQUE,
    payment_order_id BIGINT NOT NULL REFERENCES declaration_qa.business_payment_orders(id) ON DELETE CASCADE,
    receipt_no VARCHAR(120) NOT NULL UNIQUE,
    receipt_type VARCHAR(40) NOT NULL DEFAULT 'electronic_invoice',
    title VARCHAR(255) NOT NULL DEFAULT '',
    amount NUMERIC(12, 2) NOT NULL DEFAULT 0 CHECK (amount >= 0),
    status VARCHAR(40) NOT NULL DEFAULT 'issued',
    issued_at TIMESTAMPTZ NULL,
    file_name VARCHAR(255) NOT NULL DEFAULT '',
    download_status VARCHAR(40) NOT NULL DEFAULT 'unavailable',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS declaration_qa.business_submission_receipts (
    id BIGSERIAL PRIMARY KEY,
    receipt_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    receipt_no VARCHAR(120) NOT NULL UNIQUE,
    receipt_type VARCHAR(40) NOT NULL DEFAULT 'submission',
    status VARCHAR(40) NOT NULL DEFAULT 'issued',
    issued_at TIMESTAMPTZ NULL,
    file_name VARCHAR(255) NOT NULL DEFAULT '',
    download_status VARCHAR(40) NOT NULL DEFAULT 'unavailable',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_submission_receipts_apply
    ON declaration_qa.business_submission_receipts (apply_id, issued_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_result_certificates (
    id BIGSERIAL PRIMARY KEY,
    certificate_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    certificate_name VARCHAR(255) NOT NULL,
    certificate_no VARCHAR(160) NOT NULL DEFAULT '',
    certificate_type VARCHAR(40) NOT NULL DEFAULT 'electronic_result',
    status VARCHAR(40) NOT NULL DEFAULT 'processing',
    issued_at TIMESTAMPTZ NULL,
    valid_until DATE NULL,
    file_name VARCHAR(255) NOT NULL DEFAULT '',
    download_status VARCHAR(40) NOT NULL DEFAULT 'unavailable',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_result_certificates_apply
    ON declaration_qa.business_result_certificates (apply_id, issued_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_deliveries (
    id BIGSERIAL PRIMARY KEY,
    delivery_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    delivery_type VARCHAR(40) NOT NULL DEFAULT 'self_pickup',
    status VARCHAR(40) NOT NULL DEFAULT 'pending',
    courier_name VARCHAR(120) NOT NULL DEFAULT '',
    tracking_no VARCHAR(160) NOT NULL DEFAULT '',
    pickup_address TEXT NOT NULL DEFAULT '',
    shipped_at TIMESTAMPTZ NULL,
    delivered_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_deliveries_apply
    ON declaration_qa.business_deliveries (apply_id, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_consultations (
    id BIGSERIAL PRIMARY KEY,
    consultation_uid VARCHAR(120) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    consultation_type VARCHAR(40) NOT NULL DEFAULT 'consultation',
    subject VARCHAR(255) NOT NULL,
    content_summary TEXT NOT NULL DEFAULT '',
    status VARCHAR(40) NOT NULL DEFAULT 'submitted',
    channel VARCHAR(40) NOT NULL DEFAULT 'online',
    replied_at TIMESTAMPTZ NULL,
    reply_summary TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_consultations_user
    ON declaration_qa.business_consultations (user_id, created_at DESC);

INSERT INTO declaration_qa.business_appointment_slots (
    slot_uid, item_code, item_name, region_code, hall_code, hall_name, hall_address,
    service_date, start_time, end_time, capacity, reserved_count, status, payload
) VALUES
    ('SLOT-TL-ADOPTION-20260718-0900', '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '09:00', '10:00', 20, 14, 'available', jsonb_build_object('source', 'business_query_mock')),
    ('SLOT-TL-ADOPTION-20260718-1000', '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '10:00', '11:00', 20, 20, 'full', jsonb_build_object('source', 'business_query_mock')),
    ('SLOT-TL-ADOPTION-20260718-1100', '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '11:00', '12:00', 20, 10, 'available', jsonb_build_object('source', 'business_query_mock'))
ON CONFLICT (slot_uid) DO UPDATE SET
    item_name = EXCLUDED.item_name,
    hall_name = EXCLUDED.hall_name,
    hall_address = EXCLUDED.hall_address,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO declaration_qa.business_appointments (
    appointment_uid, user_id, item_code, item_name, region_code, hall_code,
    hall_name, hall_address, service_date, start_time, end_time, status,
    queue_number, people_ahead, estimated_wait_minutes, counter_name,
    checked_in_at, payload, created_at, updated_at
) VALUES
    ('APPT-USER1-20260718-001', 1, '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '09:00', '10:00', 'checked_in', 'A018', 4, 16, '综合窗口03', '2026-07-18 09:08:00+08', jsonb_build_object('source', 'business_query_mock'), '2026-07-15 14:00:00+08', '2026-07-18 09:12:00+08'),
    ('APPT-USER2-20260718-001', 2, '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '10:00', '11:00', 'booked', '', 0, 0, '', NULL, jsonb_build_object('source', 'business_query_mock'), '2026-07-15 15:00:00+08', '2026-07-15 15:00:00+08')
ON CONFLICT (appointment_uid) DO UPDATE SET
    status = EXCLUDED.status,
    queue_number = EXCLUDED.queue_number,
    people_ahead = EXCLUDED.people_ahead,
    estimated_wait_minutes = EXCLUDED.estimated_wait_minutes,
    counter_name = EXCLUDED.counter_name,
    checked_in_at = EXCLUDED.checked_in_at,
    updated_at = EXCLUDED.updated_at;

UPDATE declaration_qa.business_appointments a
SET slot_id = s.id
FROM declaration_qa.business_appointment_slots s
WHERE (a.appointment_uid = 'APPT-USER1-20260718-001' AND s.slot_uid = 'SLOT-TL-ADOPTION-20260718-0900')
   OR (a.appointment_uid = 'APPT-USER2-20260718-001' AND s.slot_uid = 'SLOT-TL-ADOPTION-20260718-1000');

WITH seed_orders (
    order_uid, apply_uid, order_no, fee_item, amount, status, payment_channel,
    payable_at, expires_at, paid_at
) AS (
    VALUES
        ('PAY-DOC40-WATER-2026', 'AHZWFW-DOC40-MOCK-WATER-2026', 'PAY-20260715-WATER', '水资源费', 550.00::numeric, 'pending', 'wanshitong', '2026-07-15 09:00:00+08'::timestamptz, '2026-07-22 23:59:59+08'::timestamptz, NULL::timestamptz),
        ('PAY-REAL40-ARCHIVE-2026', 'AHZWFW-REAL40-MOCK-005', 'PAY-20260613-ARCHIVE', '结果邮寄服务费', 12.00::numeric, 'paid', 'wanshitong', '2026-06-13 15:35:00+08'::timestamptz, '2026-06-14 23:59:59+08'::timestamptz, '2026-06-13 15:40:00+08'::timestamptz)
)
INSERT INTO declaration_qa.business_payment_orders (
    order_uid, apply_id, order_no, fee_item, amount, status, payment_channel,
    payable_at, expires_at, paid_at, payload, created_at, updated_at
)
SELECT
    s.order_uid, a.id, s.order_no, s.fee_item, s.amount, s.status, s.payment_channel,
    s.payable_at, s.expires_at, s.paid_at, jsonb_build_object('source', 'business_query_mock'),
    s.payable_at, COALESCE(s.paid_at, s.payable_at)
FROM seed_orders s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid
ON CONFLICT (order_uid) DO UPDATE SET
    status = EXCLUDED.status,
    expires_at = EXCLUDED.expires_at,
    paid_at = EXCLUDED.paid_at,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_payment_receipts (
    receipt_uid, payment_order_id, receipt_no, receipt_type, title, amount,
    status, issued_at, file_name, download_status, payload, created_at, updated_at
)
SELECT
    'PAY-RECEIPT-ARCHIVE-2026', p.id, 'DZPJ-20260613-ARCHIVE', 'electronic_invoice',
    '个人', p.amount, 'issued', '2026-06-13 15:45:00+08', '结果邮寄服务费电子票据.pdf',
    'available', jsonb_build_object('source', 'business_query_mock'),
    '2026-06-13 15:45:00+08', '2026-06-13 15:45:00+08'
FROM declaration_qa.business_payment_orders p
WHERE p.order_uid = 'PAY-REAL40-ARCHIVE-2026'
ON CONFLICT (receipt_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    updated_at = EXCLUDED.updated_at;

WITH seed_receipts (
    receipt_uid, apply_uid, receipt_no, receipt_type, issued_at, file_name
) AS (
    VALUES
        ('SUBMIT-RECEIPT-WATER-2026', 'AHZWFW-DOC40-MOCK-WATER-2026', 'TJHZ-20260710-WATER', 'submission', '2026-07-10 09:31:00+08'::timestamptz, '取水许可申请提交回执.pdf'),
        ('ACCEPT-RECEIPT-ARCHIVE-2026', 'AHZWFW-REAL40-MOCK-005', 'SLHZ-20260610-ARCHIVE', 'acceptance', '2026-06-10 10:06:00+08'::timestamptz, '档案材料收集受理回执.pdf')
)
INSERT INTO declaration_qa.business_submission_receipts (
    receipt_uid, apply_id, receipt_no, receipt_type, status, issued_at,
    file_name, download_status, payload, created_at, updated_at
)
SELECT
    s.receipt_uid, a.id, s.receipt_no, s.receipt_type, 'issued', s.issued_at,
    s.file_name, 'available', jsonb_build_object('source', 'business_query_mock'),
    s.issued_at, s.issued_at
FROM seed_receipts s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid
ON CONFLICT (receipt_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_result_certificates (
    certificate_uid, apply_id, certificate_name, certificate_no, certificate_type,
    status, issued_at, file_name, download_status, payload, created_at, updated_at
)
SELECT
    'RESULT-ARCHIVE-2026', a.id, '档案材料接收确认单', 'DAJS-2026-005',
    'electronic_result', 'issued', '2026-06-13 15:30:00+08',
    '档案材料接收确认单.pdf', 'available', jsonb_build_object('source', 'business_query_mock'),
    '2026-06-13 15:30:00+08', '2026-06-13 15:30:00+08'
FROM declaration_qa.business_applications a
WHERE a.apply_uid = 'AHZWFW-REAL40-MOCK-005'
ON CONFLICT (certificate_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_deliveries (
    delivery_uid, apply_id, delivery_type, status, courier_name, tracking_no,
    pickup_address, shipped_at, delivered_at, payload, created_at, updated_at
)
SELECT
    'DELIVERY-ARCHIVE-2026', a.id, 'mail', 'delivered', '中国邮政EMS',
    'EMS202606130005', '', '2026-06-13 17:00:00+08', '2026-06-15 14:10:00+08',
    jsonb_build_object('source', 'business_query_mock'),
    '2026-06-13 17:00:00+08', '2026-06-15 14:10:00+08'
FROM declaration_qa.business_applications a
WHERE a.apply_uid = 'AHZWFW-REAL40-MOCK-005'
ON CONFLICT (delivery_uid) DO UPDATE SET
    status = EXCLUDED.status,
    tracking_no = EXCLUDED.tracking_no,
    delivered_at = EXCLUDED.delivered_at,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_consultations (
    consultation_uid, user_id, consultation_type, subject, content_summary,
    status, channel, replied_at, reply_summary, payload, created_at, updated_at
) VALUES
    ('CONSULT-USER1-20260715-001', 1, 'consultation', '取水许可缴费期限咨询', '咨询缴费订单的支付期限和票据获取方式。', 'replied', 'online', '2026-07-15 11:20:00+08', '请在订单显示的支付期限内完成缴费，支付成功后可查询电子票据。', jsonb_build_object('source', 'business_query_mock'), '2026-07-15 10:00:00+08', '2026-07-15 11:20:00+08'),
    ('CONSULT-USER2-20260715-001', 2, 'complaint', '窗口等候时间投诉', '反映窗口等候时间较长。', 'processing', 'online', NULL, '', jsonb_build_object('source', 'business_query_mock'), '2026-07-15 09:00:00+08', '2026-07-15 09:00:00+08')
ON CONFLICT (consultation_uid) DO UPDATE SET
    status = EXCLUDED.status,
    replied_at = EXCLUDED.replied_at,
    reply_summary = EXCLUDED.reply_summary,
    updated_at = EXCLUDED.updated_at;

COMMIT;

-- ============================================================================
-- 来源：011_submitted_document_mock_appointments.sql
-- ============================================================================

-- 根据已发布政务服务材料构造的模拟服务数据。
-- 使用稳定的 source_manifest_id 值保留与整理后来源的关联。

BEGIN;

WITH seed_slots (
    slot_uid, source_manifest_id, item_code, item_name, region_code,
    hall_code, hall_name, hall_address, service_date, start_time, end_time,
    capacity, reserved_count, status
) AS (
    VALUES
        ('SLOT-DOC-WATER-20260720-0900', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-ZW-GCJS', '安徽省政务服务中心工程建设综合窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-20', TIME '09:00', TIME '10:00', 12, 5, 'available'),
        ('SLOT-DOC-WATER-20260720-1000', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-ZW-GCJS', '安徽省政务服务中心工程建设综合窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-20', TIME '10:00', TIME '11:00', 12, 12, 'full'),
        ('SLOT-DOC-WATER-20260720-1330', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-ZW-GCJS', '安徽省政务服务中心工程建设综合窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-20', TIME '13:30', TIME '14:30', 12, 2, 'available'),
        ('SLOT-DOC-WATER-SLZW-20260720-0830', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-SLT-SLZW', '安徽省政务服务中心水利窗口', '安徽省合肥市包河区马鞍山路509号二楼水利服务区', DATE '2026-07-20', TIME '08:30', TIME '09:30', 10, 3, 'available'),
        ('SLOT-DOC-WATER-SLZW-20260720-1400', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-SLT-SLZW', '安徽省政务服务中心水利窗口', '安徽省合肥市包河区马鞍山路509号二楼水利服务区', DATE '2026-07-20', TIME '14:00', TIME '15:00', 10, 4, 'available'),
        ('SLOT-DOC-WATER-HFSW-20260720-1100', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-HF-SWZW', '合肥市政务服务中心水务窗口', '安徽省合肥市蜀山区怀宁路100号三楼水务服务区', DATE '2026-07-20', TIME '11:00', TIME '12:00', 8, 2, 'available'),
        ('SLOT-DOC-WATER-HFSW-20260720-1500', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-HF-SWZW', '合肥市政务服务中心水务窗口', '安徽省合肥市蜀山区怀宁路100号三楼水务服务区', DATE '2026-07-20', TIME '15:00', TIME '16:00', 8, 1, 'available'),
        ('SLOT-DOC-GRASS-20260721-0900', 'item_30_official_ahzwfw', '11340000002985990J200016407900001', '草原防火期内在草原上进行爆破、勘察和施工等活动审批', '340000', 'AH-LYJ-ZW', '安徽省人民政府政务服务中心林业窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-21', TIME '09:00', TIME '10:00', 8, 3, 'available'),
        ('SLOT-DOC-GRASS-20260721-1400', 'item_30_official_ahzwfw', '11340000002985990J200016407900001', '草原防火期内在草原上进行爆破、勘察和施工等活动审批', '340000', 'AH-LYJ-ZW', '安徽省人民政府政务服务中心林业窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-21', TIME '14:00', TIME '15:00', 8, 7, 'available'),
        ('SLOT-DOC-SPORT-20260722-0900', 'high_risk_sports_event_permit_official_ahzwfw', '11340000002985974W234013300500001', '举办高危险性体育赛事活动许可', '340000', 'AH-TYJ-01', '安徽省体育局政务服务窗口', '安徽省合肥市包河区芜湖路97号', DATE '2026-07-22', TIME '09:00', TIME '10:00', 10, 4, 'available'),
        ('SLOT-DOC-SPORT-20260722-1500', 'high_risk_sports_event_permit_official_ahzwfw', '11340000002985974W234013300500001', '举办高危险性体育赛事活动许可', '340000', 'AH-TYJ-01', '安徽省体育局政务服务窗口', '安徽省合肥市包河区芜湖路97号', DATE '2026-07-22', TIME '15:00', TIME '16:00', 10, 10, 'full'),
        ('SLOT-DOC-MARRIAGE-20260723-0800', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记', '340000', 'AH-MZT-102', '省涉外婚姻收养登记服务中心102室', '安徽省合肥市包河区屯溪路435号安徽省民政厅102室', DATE '2026-07-23', TIME '08:00', TIME '08:30', 6, 4, 'available'),
        ('SLOT-DOC-MARRIAGE-20260723-0900', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记', '340000', 'AH-MZT-102', '省涉外婚姻收养登记服务中心102室', '安徽省合肥市包河区屯溪路435号安徽省民政厅102室', DATE '2026-07-23', TIME '09:00', TIME '09:30', 6, 1, 'available'),
        ('SLOT-DOC-MARRIAGE-20260723-1430', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记', '340000', 'AH-MZT-102', '省涉外婚姻收养登记服务中心102室', '安徽省合肥市包河区屯溪路435号安徽省民政厅102室', DATE '2026-07-23', TIME '14:30', TIME '15:00', 6, 6, 'full'),
        ('SLOT-DOC-FIRE-20260724-0830', 'fire_accident_identification_review_official_ahzwfw', '11340000MB1C86468R234070901300001', '火灾事故认定复核', '340000', 'AH-XFJY-C1', '安徽省消防救援总队政务窗口', '安徽省合肥市广西路与中山路交叉口中山路C1-104号', DATE '2026-07-24', TIME '08:30', TIME '09:30', 8, 2, 'available'),
        ('SLOT-DOC-FIRE-20260724-1430', 'fire_accident_identification_review_official_ahzwfw', '11340000MB1C86468R234070901300001', '火灾事故认定复核', '340000', 'AH-XFJY-C1', '安徽省消防救援总队政务窗口', '安徽省合肥市广西路与中山路交叉口中山路C1-104号', DATE '2026-07-24', TIME '14:30', TIME '15:30', 8, 5, 'available')
)
INSERT INTO declaration_qa.business_appointment_slots (
    slot_uid, item_code, item_name, region_code, hall_code, hall_name,
    hall_address, service_date, start_time, end_time, capacity,
    reserved_count, status, payload
)
SELECT
    s.slot_uid, s.item_code, s.item_name, s.region_code, s.hall_code, s.hall_name,
    s.hall_address, s.service_date, s.start_time, s.end_time, s.capacity,
    s.reserved_count, s.status,
    jsonb_build_object(
        'source', 'submitted_document_mock',
        'source_manifest_id', s.source_manifest_id
    )
FROM seed_slots s
ON CONFLICT (slot_uid) DO UPDATE SET
    item_code = EXCLUDED.item_code,
    item_name = EXCLUDED.item_name,
    region_code = EXCLUDED.region_code,
    hall_code = EXCLUDED.hall_code,
    hall_name = EXCLUDED.hall_name,
    hall_address = EXCLUDED.hall_address,
    service_date = EXCLUDED.service_date,
    start_time = EXCLUDED.start_time,
    end_time = EXCLUDED.end_time,
    capacity = EXCLUDED.capacity,
    payload = EXCLUDED.payload,
    updated_at = CURRENT_TIMESTAMP;

WITH seed_appointments (
    appointment_uid, slot_uid, source_manifest_id, user_id, status,
    queue_number, people_ahead, estimated_wait_minutes, counter_name,
    checked_in_at, created_at, updated_at
) AS (
    VALUES
        ('APPT-USER1-WATER-20260720-0900', 'SLOT-DOC-WATER-20260720-0900', 'item_28_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 10:00:00+08', TIMESTAMPTZ '2026-07-15 10:00:00+08'),
        ('APPT-USER1-GRASS-20260721-1400', 'SLOT-DOC-GRASS-20260721-1400', 'item_30_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 10:10:00+08', TIMESTAMPTZ '2026-07-15 10:10:00+08'),
        ('APPT-USER1-SPORT-20260722-1500', 'SLOT-DOC-SPORT-20260722-1500', 'high_risk_sports_event_permit_official_ahzwfw', 1::bigint, 'cancelled', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-14 16:00:00+08', TIMESTAMPTZ '2026-07-15 09:20:00+08'),
        ('APPT-USER1-MARRIAGE-20260723-0900', 'SLOT-DOC-MARRIAGE-20260723-0900', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 11:00:00+08', TIMESTAMPTZ '2026-07-15 11:00:00+08'),
        ('APPT-USER1-FIRE-20260724-0830', 'SLOT-DOC-FIRE-20260724-0830', 'fire_accident_identification_review_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 11:15:00+08', TIMESTAMPTZ '2026-07-15 11:15:00+08'),
        ('APPT-USER2-MARRIAGE-20260723-0800', 'SLOT-DOC-MARRIAGE-20260723-0800', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', 2::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 12:00:00+08', TIMESTAMPTZ '2026-07-15 12:00:00+08')
)
INSERT INTO declaration_qa.business_appointments (
    appointment_uid, slot_id, user_id, item_code, item_name, region_code,
    hall_code, hall_name, hall_address, service_date, start_time, end_time,
    status, queue_number, people_ahead, estimated_wait_minutes, counter_name,
    checked_in_at, payload, created_at, updated_at
)
SELECT
    a.appointment_uid, s.id, a.user_id, s.item_code, s.item_name, s.region_code,
    s.hall_code, s.hall_name, s.hall_address, s.service_date, s.start_time, s.end_time,
    a.status, a.queue_number, a.people_ahead, a.estimated_wait_minutes, a.counter_name,
    a.checked_in_at,
    jsonb_build_object(
        'source', 'submitted_document_mock',
        'source_manifest_id', a.source_manifest_id
    ),
    a.created_at, a.updated_at
FROM seed_appointments a
JOIN declaration_qa.business_appointment_slots s ON s.slot_uid = a.slot_uid
ON CONFLICT (appointment_uid) DO UPDATE SET
    slot_id = EXCLUDED.slot_id,
    item_code = EXCLUDED.item_code,
    item_name = EXCLUDED.item_name,
    region_code = EXCLUDED.region_code,
    hall_code = EXCLUDED.hall_code,
    hall_name = EXCLUDED.hall_name,
    hall_address = EXCLUDED.hall_address,
    service_date = EXCLUDED.service_date,
    start_time = EXCLUDED.start_time,
    end_time = EXCLUDED.end_time,
    status = EXCLUDED.status,
    queue_number = EXCLUDED.queue_number,
    people_ahead = EXCLUDED.people_ahead,
    estimated_wait_minutes = EXCLUDED.estimated_wait_minutes,
    counter_name = EXCLUDED.counter_name,
    checked_in_at = EXCLUDED.checked_in_at,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

WITH seed_receipts (
    receipt_uid, apply_uid, source_manifest_id, receipt_no, receipt_type,
    issued_at, file_name
) AS (
    VALUES
        ('SUBMIT-RECEIPT-SPORT-2026', 'AHZWFW-REAL40-MOCK-007', 'high_risk_sports_event_permit_official_ahzwfw', 'TJHZ-20260706-SPORT', 'submission', TIMESTAMPTZ '2026-07-06 09:31:00+08', '高危险性体育赛事活动许可申请提交回执.pdf'),
        ('ACCEPT-RECEIPT-MARRIAGE-2026', 'AHZWFW-DOC40-MOCK-MARRIAGE-2026', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', 'SLHZ-20260708-MARRIAGE', 'acceptance', TIMESTAMPTZ '2026-07-08 11:06:00+08', '涉外婚姻登记受理回执.pdf')
)
INSERT INTO declaration_qa.business_submission_receipts (
    receipt_uid, apply_id, receipt_no, receipt_type, status, issued_at,
    file_name, download_status, payload, created_at, updated_at
)
SELECT
    r.receipt_uid, a.id, r.receipt_no, r.receipt_type, 'issued', r.issued_at,
    r.file_name, 'available',
    jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', r.source_manifest_id),
    r.issued_at, r.issued_at
FROM seed_receipts r
JOIN declaration_qa.business_applications a ON a.apply_uid = r.apply_uid
ON CONFLICT (receipt_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

WITH seed_certificates (
    certificate_uid, apply_uid, source_manifest_id, certificate_name,
    certificate_no, issued_at, file_name
) AS (
    VALUES
        ('RESULT-ADOPTION-2025', 'AHZWFW-DOC40-MOCK-ADOPTION-2025', 'foreign_related_adoption_registration_official_ahzwfw', '涉外收养登记证', 'AH-SY-2025-0818', TIMESTAMPTZ '2025-08-18 15:10:00+08', '涉外收养登记证.pdf'),
        ('RESULT-OCS-2025', 'AHZWFW-DOC40-MOCK-OCS-2025', 'overseas_chinese_return_settlement_approval_official_ahzwfw', '华侨回国定居证', 'AH-HQDJ-2025-0320', TIMESTAMPTZ '2025-03-20 16:30:00+08', '华侨回国定居证.pdf')
)
INSERT INTO declaration_qa.business_result_certificates (
    certificate_uid, apply_id, certificate_name, certificate_no,
    certificate_type, status, issued_at, file_name, download_status,
    payload, created_at, updated_at
)
SELECT
    c.certificate_uid, a.id, c.certificate_name, c.certificate_no,
    'electronic_certificate', 'issued', c.issued_at, c.file_name, 'available',
    jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', c.source_manifest_id),
    c.issued_at, c.issued_at
FROM seed_certificates c
JOIN declaration_qa.business_applications a ON a.apply_uid = c.apply_uid
ON CONFLICT (certificate_uid) DO UPDATE SET
    status = EXCLUDED.status,
    file_name = EXCLUDED.file_name,
    download_status = EXCLUDED.download_status,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

WITH seed_deliveries (
    delivery_uid, apply_uid, source_manifest_id, delivery_type, status,
    courier_name, tracking_no, pickup_address, shipped_at, delivered_at,
    created_at, updated_at
) AS (
    VALUES
        ('DELIVERY-ADOPTION-2025', 'AHZWFW-DOC40-MOCK-ADOPTION-2025', 'foreign_related_adoption_registration_official_ahzwfw', 'self_pickup', 'ready_for_pickup', '', '', '安徽省合肥市包河区屯溪路435号安徽省民政厅123室', NULL::timestamptz, NULL::timestamptz, TIMESTAMPTZ '2025-08-18 15:15:00+08', TIMESTAMPTZ '2025-08-18 15:15:00+08'),
        ('DELIVERY-OCS-2025', 'AHZWFW-DOC40-MOCK-OCS-2025', 'overseas_chinese_return_settlement_approval_official_ahzwfw', 'mail', 'delivered', '中国邮政EMS', 'EMS202503200013', '', TIMESTAMPTZ '2025-03-20 17:00:00+08', TIMESTAMPTZ '2025-03-22 10:30:00+08', TIMESTAMPTZ '2025-03-20 17:00:00+08', TIMESTAMPTZ '2025-03-22 10:30:00+08')
)
INSERT INTO declaration_qa.business_deliveries (
    delivery_uid, apply_id, delivery_type, status, courier_name, tracking_no,
    pickup_address, shipped_at, delivered_at, payload, created_at, updated_at
)
SELECT
    d.delivery_uid, a.id, d.delivery_type, d.status, d.courier_name, d.tracking_no,
    d.pickup_address, d.shipped_at, d.delivered_at,
    jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', d.source_manifest_id),
    d.created_at, d.updated_at
FROM seed_deliveries d
JOIN declaration_qa.business_applications a ON a.apply_uid = d.apply_uid
ON CONFLICT (delivery_uid) DO UPDATE SET
    status = EXCLUDED.status,
    tracking_no = EXCLUDED.tracking_no,
    pickup_address = EXCLUDED.pickup_address,
    shipped_at = EXCLUDED.shipped_at,
    delivered_at = EXCLUDED.delivered_at,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_consultations (
    consultation_uid, user_id, consultation_type, subject, content_summary,
    status, channel, replied_at, reply_summary, payload, created_at, updated_at
) VALUES
    ('CONSULT-USER1-WATER-20260715', 1, 'consultation', '取水许可现场预约咨询', '咨询取水许可是否可以预约工程建设综合窗口。', 'replied', 'online', TIMESTAMPTZ '2026-07-15 09:20:00+08', '可以预约马鞍山路509号工程建设综合窗口，请按预约时段携带核验材料。', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'item_28_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 09:00:00+08', TIMESTAMPTZ '2026-07-15 09:20:00+08'),
    ('CONSULT-USER1-SPORT-20260715', 1, 'consultation', '高危险性体育赛事材料咨询', '咨询赛事活动许可的场地和安全保障材料。', 'processing', 'online', NULL, '', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'high_risk_sports_event_permit_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 10:30:00+08', TIMESTAMPTZ '2026-07-15 10:30:00+08'),
    ('CONSULT-USER1-MARRIAGE-20260715', 1, 'consultation', '涉外婚姻登记预约咨询', '咨询双方是否需要共同到场以及可预约时段。', 'replied', 'online', TIMESTAMPTZ '2026-07-15 11:40:00+08', '该事项为窗口办理，请双方按预约时间共同前往省涉外婚姻收养登记服务中心。', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 11:10:00+08', TIMESTAMPTZ '2026-07-15 11:40:00+08'),
    ('CONSULT-USER2-FIRE-20260715', 2, 'consultation', '火灾事故认定复核办理地点咨询', '咨询火灾事故认定复核的线下办理地点。', 'replied', 'online', TIMESTAMPTZ '2026-07-15 14:20:00+08', '办理地点为中山路C1-104号安徽省消防救援总队政务窗口。', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'fire_accident_identification_review_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 14:00:00+08', TIMESTAMPTZ '2026-07-15 14:20:00+08')
ON CONFLICT (consultation_uid) DO UPDATE SET
    status = EXCLUDED.status,
    replied_at = EXCLUDED.replied_at,
    reply_summary = EXCLUDED.reply_summary,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

COMMIT;

BEGIN;

ALTER TABLE declaration_qa.business_consultations
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(120) NOT NULL DEFAULT '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_consultations_idempotency
    ON declaration_qa.business_consultations (user_id, idempotency_key)
    WHERE idempotency_key <> '';

COMMIT;

-- ============================================================================
-- 来源：015_water_permit_application_materials.sql
-- ============================================================================

-- Publish the water-permit v2 schema with mandatory application-material uploads.

BEGIN;

UPDATE declaration_qa.business_item_form_schemas
SET status = 'deprecated',
    updated_at = CURRENT_TIMESTAMP
WHERE item_code = '11340000002985923L200011900100001'
  AND region_code = ''
  AND version_no < 2;

INSERT INTO declaration_qa.business_item_form_schemas (
    schema_uid, item_code, item_name, biz_type, region_code, version_no, status, fields, metadata
) VALUES (
    'water-permit-v2',
    '11340000002985923L200011900100001',
    '取水许可',
    'water_permit',
    '',
    2,
    'published',
    '[
      {"name":"applicant_name","label":"申请单位或个人","type":"text","required":true,"max_length":200,"aliases":["申请人","单位名称"]},
      {"name":"credit_code","label":"统一社会信用代码或证件号码","type":"text","required":true,"max_length":40,"aliases":["统一社会信用代码","证件号码"]},
      {"name":"contact_name","label":"联系人","type":"text","required":true,"max_length":80,"aliases":["经办人"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话"]},
      {"name":"water_source","label":"取水水源","type":"text","required":true,"max_length":200,"aliases":["水源"]},
      {"name":"annual_water_volume","label":"年取水量（立方米）","type":"number","required":true,"aliases":["年取水量","取水量"]},
      {"name":"water_use","label":"取水用途","type":"multiselect","required":true,"options":["制水供水","原水供水","生活用水","工业用水","农业用水","其他"],"aliases":["用途"]},
      {
        "name":"application_materials",
        "label":"申报材料",
        "type":"material_group",
        "required":true,
        "description":"支持 PDF、Word、Excel、JPG、PNG、ZIP；单个文件不超过 20 MB。",
        "accept":[".pdf",".doc",".docx",".xls",".xlsx",".jpg",".jpeg",".png",".zip"],
        "materials":[
          {"key":"water_resource_report","label":"建设项目水资源论证报告书（表）","required":true,"max_files":3},
          {"key":"water_right_transfer_report","label":"水权转让可行性研究报告","required":false,"description":"通过水权转让方式获得取水指标时上传","max_files":3},
          {"key":"filing_materials","label":"备案材料","required":false,"description":"属于备案项目时上传","max_files":3},
          {"key":"business_license_or_id","label":"营业执照或身份证","required":true,"max_files":2},
          {"key":"water_permit_application","label":"取水许可申请书","required":true,"max_files":2},
          {"key":"interested_party_statement","label":"利害关系说明","required":true,"max_files":3}
        ]
      }
    ]'::jsonb,
    jsonb_build_object(
        'source', 'water_permit_application_materials',
        'source_item_code', '11340000002985923L200011900100001'
    )
)
ON CONFLICT (item_code, region_code, version_no) DO UPDATE SET
    schema_uid = EXCLUDED.schema_uid,
    item_name = EXCLUDED.item_name,
    biz_type = EXCLUDED.biz_type,
    status = EXCLUDED.status,
    fields = EXCLUDED.fields,
    metadata = EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

COMMIT;
