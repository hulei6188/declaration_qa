-- Agentic ticket tools require an item-scoped application query and a complete,
-- stable status timeline for every application. This migration is idempotent.

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
    ADD COLUMN IF NOT EXISTS tool_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS agent_iteration INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS agent_selected_item_code VARCHAR(160) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS agent_selected_application_id BIGINT NULL;

CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_tool_results_gin
    ON declaration_qa.audit_logs USING GIN (tool_results);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_audit_logs_agent_item
    ON declaration_qa.audit_logs (agent_selected_item_code, created_at DESC);

COMMIT;
