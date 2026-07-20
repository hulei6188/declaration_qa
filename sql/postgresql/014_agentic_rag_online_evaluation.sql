BEGIN;

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
    ON declaration_qa.eval_cases (source_trace_id)
    WHERE source_trace_id <> '';

CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_candidates_trace
    ON declaration_qa.eval_candidates (trace_id)
    WHERE trace_id <> '';

COMMIT;
