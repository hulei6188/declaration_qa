ALTER TABLE declaration_qa.eval_cases
    ADD COLUMN IF NOT EXISTS reference_answer TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE declaration_qa.eval_run_items
    ADD COLUMN IF NOT EXISTS reference_answer TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS reference_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS retrieved_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS ragas_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS ragas_status JSONB NOT NULL DEFAULT '{}'::jsonb;
