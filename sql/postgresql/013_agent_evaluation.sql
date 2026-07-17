BEGIN;

ALTER TABLE declaration_qa.eval_datasets
    ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'draft',
    ADD COLUMN IF NOT EXISTS source VARCHAR(40) NOT NULL DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS parent_dataset_id BIGINT NULL REFERENCES declaration_qa.eval_datasets(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS knowledge_revision VARCHAR(160) NOT NULL DEFAULT '';

ALTER TABLE declaration_qa.eval_cases
    ADD COLUMN IF NOT EXISTS reference_answer TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
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
    ADD COLUMN IF NOT EXISTS reference_answer TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS retrieved_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS ragas_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS ragas_status JSONB NOT NULL DEFAULT '{}'::jsonb,
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

CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_datasets_status
    ON declaration_qa.eval_datasets (status, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_cases_type
    ON declaration_qa.eval_cases (dataset_id, case_type, enabled);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_runs_baseline
    ON declaration_qa.eval_runs (baseline_run_id) WHERE baseline_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_run_items_passed
    ON declaration_qa.eval_run_items (run_id, passed);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_candidates_status
    ON declaration_qa.eval_candidates (status, severity, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_candidates_audit
    ON declaration_qa.eval_candidates (audit_uid) WHERE audit_uid <> '';
CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_scores_item
    ON declaration_qa.eval_scores (run_item_id, metric_name);

COMMIT;
