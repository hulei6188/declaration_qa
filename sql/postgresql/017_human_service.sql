BEGIN;

CREATE TABLE IF NOT EXISTS declaration_qa.human_service_tickets (
    id BIGSERIAL PRIMARY KEY,
    ticket_uid VARCHAR(80) NOT NULL UNIQUE,
    audit_uid VARCHAR(120) NOT NULL DEFAULT '',
    session_id VARCHAR(120) NOT NULL DEFAULT '',
    user_id BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    priority VARCHAR(20) NOT NULL DEFAULT 'normal',
    assigned_to BIGINT NULL,
    query TEXT NOT NULL DEFAULT '',
    intent VARCHAR(120) NOT NULL DEFAULT '',
    route VARCHAR(120) NOT NULL DEFAULT '',
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS declaration_qa.human_service_messages (
    id BIGSERIAL PRIMARY KEY,
    ticket_id BIGINT NOT NULL REFERENCES declaration_qa.human_service_tickets(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    author_id BIGINT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_declaration_qa_human_service_status ON declaration_qa.human_service_tickets (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_human_service_user ON declaration_qa.human_service_tickets (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_human_service_messages_ticket ON declaration_qa.human_service_messages (ticket_id, created_at ASC);

COMMIT;
