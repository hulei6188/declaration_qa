-- Persist a replayable Agent run without exposing internal messages in normal chat history.
-- This migration is idempotent and preserves all existing session messages.

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
