-- Isolated schema for the Declaration QA Workflow service.
--
-- This file is intentionally separate from AgentBase's main schema. The
-- declaration QA service starts from declaration_qa.api.server.app:app and stores its
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

CREATE TABLE IF NOT EXISTS declaration_qa.business_applications (
    id BIGSERIAL PRIMARY KEY,
    apply_uid VARCHAR(120) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
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

CREATE TABLE IF NOT EXISTS declaration_qa.business_application_statuses (
    id BIGSERIAL PRIMARY KEY,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    status VARCHAR(40) NOT NULL,
    current_node VARCHAR(80) NOT NULL DEFAULT '',
    node_name VARCHAR(120) NOT NULL DEFAULT '',
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_business_status_current
    ON declaration_qa.business_application_statuses (apply_id)
    WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_business_status_apply
    ON declaration_qa.business_application_statuses (apply_id, created_at DESC);

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
    retrieval_chain VARCHAR(40) NOT NULL DEFAULT '',
    region_code VARCHAR(20) NOT NULL,
    biz_type VARCHAR(80) NULL,
    apply_id VARCHAR(80) NOT NULL DEFAULT '',
    business_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS retrieval_chain VARCHAR(40) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS business_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE declaration_qa.audit_logs
    ADD COLUMN IF NOT EXISTS evidence_decision JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_user
    ON declaration_qa.audit_logs (user_id, created_at DESC);
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
COMMENT ON COLUMN declaration_qa.business_application_statuses.status IS '状态编码';
COMMENT ON COLUMN declaration_qa.business_application_statuses.current_node IS '审核节点编码';
COMMENT ON COLUMN declaration_qa.business_application_statuses.node_name IS '审核节点名称';
COMMENT ON COLUMN declaration_qa.business_application_statuses.is_current IS '是否当前有效状态';
COMMENT ON COLUMN declaration_qa.business_application_statuses.payload IS '状态原始扩展数据';
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
COMMENT ON COLUMN declaration_qa.messages.role IS '消息角色：user/assistant/system';
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
COMMENT ON COLUMN declaration_qa.audit_logs.retrieval_chain IS '回答链路：knowledge_retrieval/ticket_augmented';
COMMENT ON COLUMN declaration_qa.audit_logs.region_code IS '行政区划编码';
COMMENT ON COLUMN declaration_qa.audit_logs.biz_type IS '业务类型，仅用于审计业务上下文';
COMMENT ON COLUMN declaration_qa.audit_logs.apply_id IS '申请单 ID';
COMMENT ON COLUMN declaration_qa.audit_logs.business_snapshot IS '真实工单业务快照';
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

COMMIT;
