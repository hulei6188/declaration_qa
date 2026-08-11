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
    ADD COLUMN IF NOT EXISTS online_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS attribution_category VARCHAR(40) NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS attribution_reason TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS knowledge_draft JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS fingerprint VARCHAR(64) NOT NULL DEFAULT '';

-- Preserve historical duplicate candidates, but assign the active dedupe key
-- only to the newest row for each stable sampling fingerprint.
WITH ranked AS (
    SELECT
        id,
        COALESCE(NULLIF(detection_signals ->> 'fingerprint', ''), '') AS detected_fingerprint,
        row_number() OVER (
            PARTITION BY detection_signals ->> 'fingerprint'
            ORDER BY created_at DESC, id DESC
        ) AS fingerprint_rank
    FROM declaration_qa.eval_candidates
    WHERE fingerprint = ''
      AND COALESCE(NULLIF(detection_signals ->> 'fingerprint', ''), '') <> ''
      AND status IN ('new', 'triaged', 'labeled')
      AND NOT EXISTS (
          SELECT 1
          FROM declaration_qa.eval_candidates AS keyed
          WHERE keyed.fingerprint = eval_candidates.detection_signals ->> 'fingerprint'
            AND keyed.status IN ('new', 'triaged', 'labeled')
      )
)
UPDATE declaration_qa.eval_candidates AS candidates
SET fingerprint = ranked.detected_fingerprint
FROM ranked
WHERE candidates.id = ranked.id
  AND ranked.fingerprint_rank = 1;

CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_cases_source_trace
    ON declaration_qa.eval_cases (source_trace_id)
    WHERE source_trace_id <> '';

CREATE INDEX IF NOT EXISTS ix_declaration_qa_eval_candidates_trace
    ON declaration_qa.eval_candidates (trace_id)
    WHERE trace_id <> '';

CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_eval_candidates_active_fingerprint
    ON declaration_qa.eval_candidates (fingerprint)
    WHERE fingerprint <> '' AND status IN ('new', 'triaged', 'labeled');

COMMIT;
