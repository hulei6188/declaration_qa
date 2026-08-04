-- Versioned item schemas and idempotent write support for multi-turn form filling.

BEGIN;

CREATE TABLE IF NOT EXISTS declaration_qa.business_item_form_schemas (
    id BIGSERIAL PRIMARY KEY,
    schema_uid VARCHAR(160) NOT NULL UNIQUE,
    item_code VARCHAR(160) NOT NULL,
    item_name VARCHAR(255) NOT NULL,
    biz_type VARCHAR(80) NOT NULL DEFAULT 'application',
    region_code VARCHAR(20) NOT NULL DEFAULT '',
    version_no INTEGER NOT NULL DEFAULT 1 CHECK (version_no > 0),
    status VARCHAR(40) NOT NULL DEFAULT 'draft',
    fields JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(fields) = 'array'),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (item_code, region_code, version_no)
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_item_form_schemas_lookup
    ON declaration_qa.business_item_form_schemas (item_code, region_code, status, version_no DESC);

ALTER TABLE declaration_qa.business_applications
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(120) NOT NULL DEFAULT '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_applications_idempotency
    ON declaration_qa.business_applications (user_id, idempotency_key)
    WHERE idempotency_key <> '';

ALTER TABLE declaration_qa.business_consultations
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(120) NOT NULL DEFAULT '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_consultations_idempotency
    ON declaration_qa.business_consultations (user_id, idempotency_key)
    WHERE idempotency_key <> '';

INSERT INTO declaration_qa.business_item_form_schemas (
    schema_uid, item_code, item_name, biz_type, region_code, version_no, status, fields, metadata
) VALUES
(
    'residence-permit-v1', 'ITEM-RESIDENCE-PERMIT', '居住证办理', 'residence_permit', '', 1, 'published',
    '[
      {"name":"applicant_name","label":"申请人姓名","type":"text","required":true,"max_length":80,"aliases":["姓名","申请人"]},
      {"name":"id_number","label":"证件号码","type":"id_number","required":true,"aliases":["身份证号","身份证号码"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},
      {"name":"residence_address","label":"现居住地址","type":"text","required":true,"max_length":300,"aliases":["住址","居住地址"]},
      {"name":"residence_proof_no","label":"居住证明编号","type":"text","required":true,"max_length":100,"aliases":["证明编号"]}
    ]'::jsonb,
    jsonb_build_object('source', 'multiturn_form_filling_seed')
),
(
    'health-certificate-v1', 'ITEM-HEALTH-CERTIFICATE', '健康证办理', 'health_certificate', '', 1, 'published',
    '[
      {"name":"applicant_name","label":"申请人姓名","type":"text","required":true,"max_length":80,"aliases":["姓名","申请人"]},
      {"name":"id_number","label":"证件号码","type":"id_number","required":true,"aliases":["身份证号","身份证号码"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},
      {"name":"employer_name","label":"从业单位","type":"text","required":true,"max_length":200,"aliases":["单位","工作单位"]},
      {"name":"occupation_type","label":"从业类别","type":"select","required":true,"options":["食品","公共场所","饮用水","化妆品","其他"],"aliases":["职业类别","行业"]}
    ]'::jsonb,
    jsonb_build_object('source', 'multiturn_form_filling_seed')
),
(
    'water-permit-v1', '11340000002985923L200011900100001', '取水许可', 'water_permit', '', 1, 'published',
    '[
      {"name":"applicant_name","label":"申请单位或个人","type":"text","required":true,"max_length":200,"aliases":["申请人","单位名称"]},
      {"name":"credit_code","label":"统一社会信用代码或证件号码","type":"text","required":true,"max_length":40,"aliases":["统一社会信用代码","证件号码"]},
      {"name":"contact_name","label":"联系人","type":"text","required":true,"max_length":80,"aliases":["经办人"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话"]},
      {"name":"water_source","label":"取水水源","type":"text","required":true,"max_length":200,"aliases":["水源"]},
      {"name":"annual_water_volume","label":"年取水量（立方米）","type":"number","required":true,"aliases":["年取水量","取水量"]},
      {"name":"water_use","label":"取水用途","type":"multiselect","required":true,"options":["制水供水","原水供水","生活用水","工业用水","农业用水","其他"],"aliases":["用途"]}
    ]'::jsonb,
    jsonb_build_object('source', 'multiturn_form_filling_seed')
)
ON CONFLICT (schema_uid) DO UPDATE SET
    item_name = EXCLUDED.item_name,
    biz_type = EXCLUDED.biz_type,
    status = EXCLUDED.status,
    fields = EXCLUDED.fields,
    metadata = EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

COMMIT;
