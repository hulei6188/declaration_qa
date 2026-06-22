-- Fix business ticket payloads where payload.item_code accidentally stores
-- the curated document manifest id (for example *_official_ahzwfw).
--
-- After this migration:
--   payload.item_code           = official government item code used by chunks.metadata.item_code
--   payload.source_manifest_id  = curated document manifest id
--   payload.source_file         = source markdown filename when it can be resolved
--
-- The script is idempotent and only touches rows whose current item_code is a
-- manifest-style id.

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
