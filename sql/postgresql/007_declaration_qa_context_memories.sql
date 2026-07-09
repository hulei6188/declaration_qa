-- 为申报问答新增长期上下文记忆存储。

BEGIN;

CREATE TABLE IF NOT EXISTS declaration_qa.context_memories (
    id BIGSERIAL PRIMARY KEY,
    memory_uid VARCHAR(120) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    session_uid VARCHAR(80) NOT NULL DEFAULT '',
    memory_type VARCHAR(40) NOT NULL DEFAULT 'qa_fact',
    content TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    source_audit_uid VARCHAR(80) NOT NULL DEFAULT '',
    source_query TEXT NOT NULL DEFAULT '',
    source_answer TEXT NOT NULL DEFAULT '',
    region_code VARCHAR(20) NOT NULL DEFAULT '',
    biz_type VARCHAR(80) NOT NULL DEFAULT '',
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    status VARCHAR(20) NOT NULL DEFAULT 'published' CHECK (status IN ('published', 'deprecated')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_declaration_qa_context_memories_user
    ON declaration_qa.context_memories (user_id, updated_at DESC)
    WHERE status = 'published';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_context_memories_session
    ON declaration_qa.context_memories (session_uid, updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_context_memories_metadata_gin
    ON declaration_qa.context_memories USING GIN (metadata);

COMMENT ON TABLE declaration_qa.context_memories IS '长期上下文记忆表，保存每轮问答后异步抽取的可检索记忆';
COMMENT ON COLUMN declaration_qa.context_memories.memory_uid IS '长期记忆唯一标识';
COMMENT ON COLUMN declaration_qa.context_memories.user_id IS '用户 ID，用于强制隔离长期记忆';
COMMENT ON COLUMN declaration_qa.context_memories.content IS '写入向量和全文索引的长期记忆内容';
COMMENT ON COLUMN declaration_qa.context_memories.source_audit_uid IS '来源审计日志唯一标识';

COMMIT;
