-- Business data used by appointment booking and read-only service query tools.

BEGIN;

CREATE TABLE IF NOT EXISTS declaration_qa.business_appointment_slots (
    id BIGSERIAL PRIMARY KEY,
    slot_uid VARCHAR(120) NOT NULL UNIQUE,
    item_code VARCHAR(160) NOT NULL,
    item_name VARCHAR(255) NOT NULL DEFAULT '',
    region_code VARCHAR(20) NOT NULL DEFAULT '340000',
    hall_code VARCHAR(80) NOT NULL,
    hall_name VARCHAR(255) NOT NULL,
    hall_address TEXT NOT NULL DEFAULT '',
    service_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    capacity INTEGER NOT NULL DEFAULT 0 CHECK (capacity >= 0),
    reserved_count INTEGER NOT NULL DEFAULT 0 CHECK (reserved_count >= 0),
    status VARCHAR(40) NOT NULL DEFAULT 'available',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_appointment_slots_lookup
    ON declaration_qa.business_appointment_slots (item_code, region_code, service_date, hall_code);

CREATE TABLE IF NOT EXISTS declaration_qa.business_appointments (
    id BIGSERIAL PRIMARY KEY,
    appointment_uid VARCHAR(120) NOT NULL UNIQUE,
    slot_id BIGINT NULL REFERENCES declaration_qa.business_appointment_slots(id) ON DELETE RESTRICT,
    idempotency_key VARCHAR(120) NOT NULL DEFAULT '',
    user_id BIGINT NOT NULL,
    item_code VARCHAR(160) NOT NULL,
    item_name VARCHAR(255) NOT NULL DEFAULT '',
    region_code VARCHAR(20) NOT NULL DEFAULT '340000',
    hall_code VARCHAR(80) NOT NULL,
    hall_name VARCHAR(255) NOT NULL,
    hall_address TEXT NOT NULL DEFAULT '',
    service_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    status VARCHAR(40) NOT NULL DEFAULT 'booked',
    queue_number VARCHAR(40) NOT NULL DEFAULT '',
    people_ahead INTEGER NOT NULL DEFAULT 0 CHECK (people_ahead >= 0),
    estimated_wait_minutes INTEGER NOT NULL DEFAULT 0 CHECK (estimated_wait_minutes >= 0),
    counter_name VARCHAR(120) NOT NULL DEFAULT '',
    checked_in_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_appointments_user
    ON declaration_qa.business_appointments (user_id, service_date DESC, updated_at DESC);
ALTER TABLE declaration_qa.business_appointment_slots
    ADD COLUMN IF NOT EXISTS item_name VARCHAR(255) NOT NULL DEFAULT '';
ALTER TABLE declaration_qa.business_appointments
    ADD COLUMN IF NOT EXISTS slot_id BIGINT NULL REFERENCES declaration_qa.business_appointment_slots(id) ON DELETE RESTRICT;
ALTER TABLE declaration_qa.business_appointments
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(120) NOT NULL DEFAULT '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_appointments_idempotency
    ON declaration_qa.business_appointments (user_id, idempotency_key)
    WHERE idempotency_key <> '';
CREATE UNIQUE INDEX IF NOT EXISTS ux_declaration_qa_appointments_active_slot
    ON declaration_qa.business_appointments (user_id, slot_id)
    WHERE slot_id IS NOT NULL AND status IN ('booked', 'checked_in', 'waiting', 'called');

CREATE TABLE IF NOT EXISTS declaration_qa.business_payment_orders (
    id BIGSERIAL PRIMARY KEY,
    order_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    order_no VARCHAR(120) NOT NULL UNIQUE,
    fee_item VARCHAR(255) NOT NULL,
    amount NUMERIC(12, 2) NOT NULL DEFAULT 0 CHECK (amount >= 0),
    currency VARCHAR(12) NOT NULL DEFAULT 'CNY',
    status VARCHAR(40) NOT NULL DEFAULT 'pending',
    payment_channel VARCHAR(80) NOT NULL DEFAULT '',
    payable_at TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NULL,
    paid_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_payment_orders_apply
    ON declaration_qa.business_payment_orders (apply_id, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_payment_receipts (
    id BIGSERIAL PRIMARY KEY,
    receipt_uid VARCHAR(120) NOT NULL UNIQUE,
    payment_order_id BIGINT NOT NULL REFERENCES declaration_qa.business_payment_orders(id) ON DELETE CASCADE,
    receipt_no VARCHAR(120) NOT NULL UNIQUE,
    receipt_type VARCHAR(40) NOT NULL DEFAULT 'electronic_invoice',
    title VARCHAR(255) NOT NULL DEFAULT '',
    amount NUMERIC(12, 2) NOT NULL DEFAULT 0 CHECK (amount >= 0),
    status VARCHAR(40) NOT NULL DEFAULT 'issued',
    issued_at TIMESTAMPTZ NULL,
    file_name VARCHAR(255) NOT NULL DEFAULT '',
    download_status VARCHAR(40) NOT NULL DEFAULT 'unavailable',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS declaration_qa.business_submission_receipts (
    id BIGSERIAL PRIMARY KEY,
    receipt_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    receipt_no VARCHAR(120) NOT NULL UNIQUE,
    receipt_type VARCHAR(40) NOT NULL DEFAULT 'submission',
    status VARCHAR(40) NOT NULL DEFAULT 'issued',
    issued_at TIMESTAMPTZ NULL,
    file_name VARCHAR(255) NOT NULL DEFAULT '',
    download_status VARCHAR(40) NOT NULL DEFAULT 'unavailable',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_submission_receipts_apply
    ON declaration_qa.business_submission_receipts (apply_id, issued_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_result_certificates (
    id BIGSERIAL PRIMARY KEY,
    certificate_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    certificate_name VARCHAR(255) NOT NULL,
    certificate_no VARCHAR(160) NOT NULL DEFAULT '',
    certificate_type VARCHAR(40) NOT NULL DEFAULT 'electronic_result',
    status VARCHAR(40) NOT NULL DEFAULT 'processing',
    issued_at TIMESTAMPTZ NULL,
    valid_until DATE NULL,
    file_name VARCHAR(255) NOT NULL DEFAULT '',
    download_status VARCHAR(40) NOT NULL DEFAULT 'unavailable',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_result_certificates_apply
    ON declaration_qa.business_result_certificates (apply_id, issued_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_deliveries (
    id BIGSERIAL PRIMARY KEY,
    delivery_uid VARCHAR(120) NOT NULL UNIQUE,
    apply_id BIGINT NOT NULL REFERENCES declaration_qa.business_applications(id) ON DELETE CASCADE,
    delivery_type VARCHAR(40) NOT NULL DEFAULT 'self_pickup',
    status VARCHAR(40) NOT NULL DEFAULT 'pending',
    courier_name VARCHAR(120) NOT NULL DEFAULT '',
    tracking_no VARCHAR(160) NOT NULL DEFAULT '',
    pickup_address TEXT NOT NULL DEFAULT '',
    shipped_at TIMESTAMPTZ NULL,
    delivered_at TIMESTAMPTZ NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_deliveries_apply
    ON declaration_qa.business_deliveries (apply_id, created_at DESC);

CREATE TABLE IF NOT EXISTS declaration_qa.business_consultations (
    id BIGSERIAL PRIMARY KEY,
    consultation_uid VARCHAR(120) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL,
    consultation_type VARCHAR(40) NOT NULL DEFAULT 'consultation',
    subject VARCHAR(255) NOT NULL,
    content_summary TEXT NOT NULL DEFAULT '',
    status VARCHAR(40) NOT NULL DEFAULT 'submitted',
    channel VARCHAR(40) NOT NULL DEFAULT 'online',
    replied_at TIMESTAMPTZ NULL,
    reply_summary TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_declaration_qa_consultations_user
    ON declaration_qa.business_consultations (user_id, created_at DESC);

INSERT INTO declaration_qa.business_appointment_slots (
    slot_uid, item_code, item_name, region_code, hall_code, hall_name, hall_address,
    service_date, start_time, end_time, capacity, reserved_count, status, payload
) VALUES
    ('SLOT-TL-ADOPTION-20260718-0900', '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '09:00', '10:00', 20, 14, 'available', jsonb_build_object('source', 'business_query_mock')),
    ('SLOT-TL-ADOPTION-20260718-1000', '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '10:00', '11:00', 20, 20, 'full', jsonb_build_object('source', 'business_query_mock')),
    ('SLOT-TL-ADOPTION-20260718-1100', '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '11:00', '12:00', 20, 10, 'available', jsonb_build_object('source', 'business_query_mock'))
ON CONFLICT (slot_uid) DO UPDATE SET
    item_name = EXCLUDED.item_name,
    hall_name = EXCLUDED.hall_name,
    hall_address = EXCLUDED.hall_address,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO declaration_qa.business_appointments (
    appointment_uid, user_id, item_code, item_name, region_code, hall_code,
    hall_name, hall_address, service_date, start_time, end_time, status,
    queue_number, people_ahead, estimated_wait_minutes, counter_name,
    checked_in_at, payload, created_at, updated_at
) VALUES
    ('APPT-USER1-20260718-001', 1, '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '09:00', '10:00', 'checked_in', 'A018', 4, 16, '综合窗口03', '2026-07-18 09:08:00+08', jsonb_build_object('source', 'business_query_mock'), '2026-07-15 14:00:00+08', '2026-07-18 09:12:00+08'),
    ('APPT-USER2-20260718-001', 2, '11340000002986424J200071100200001', '涉外收养登记', '340000', 'AH-ZW-01', '安徽省政务服务中心', '马鞍山路509号', '2026-07-18', '10:00', '11:00', 'booked', '', 0, 0, '', NULL, jsonb_build_object('source', 'business_query_mock'), '2026-07-15 15:00:00+08', '2026-07-15 15:00:00+08')
ON CONFLICT (appointment_uid) DO UPDATE SET
    status = EXCLUDED.status,
    queue_number = EXCLUDED.queue_number,
    people_ahead = EXCLUDED.people_ahead,
    estimated_wait_minutes = EXCLUDED.estimated_wait_minutes,
    counter_name = EXCLUDED.counter_name,
    checked_in_at = EXCLUDED.checked_in_at,
    updated_at = EXCLUDED.updated_at;

UPDATE declaration_qa.business_appointments a
SET slot_id = s.id
FROM declaration_qa.business_appointment_slots s
WHERE (a.appointment_uid = 'APPT-USER1-20260718-001' AND s.slot_uid = 'SLOT-TL-ADOPTION-20260718-0900')
   OR (a.appointment_uid = 'APPT-USER2-20260718-001' AND s.slot_uid = 'SLOT-TL-ADOPTION-20260718-1000');

WITH seed_orders (
    order_uid, apply_uid, order_no, fee_item, amount, status, payment_channel,
    payable_at, expires_at, paid_at
) AS (
    VALUES
        ('PAY-DOC40-WATER-2026', 'AHZWFW-DOC40-MOCK-WATER-2026', 'PAY-20260715-WATER', '水资源费', 550.00::numeric, 'pending', 'wanshitong', '2026-07-15 09:00:00+08'::timestamptz, '2026-07-22 23:59:59+08'::timestamptz, NULL::timestamptz),
        ('PAY-REAL40-ARCHIVE-2026', 'AHZWFW-REAL40-MOCK-005', 'PAY-20260613-ARCHIVE', '结果邮寄服务费', 12.00::numeric, 'paid', 'wanshitong', '2026-06-13 15:35:00+08'::timestamptz, '2026-06-14 23:59:59+08'::timestamptz, '2026-06-13 15:40:00+08'::timestamptz)
)
INSERT INTO declaration_qa.business_payment_orders (
    order_uid, apply_id, order_no, fee_item, amount, status, payment_channel,
    payable_at, expires_at, paid_at, payload, created_at, updated_at
)
SELECT
    s.order_uid, a.id, s.order_no, s.fee_item, s.amount, s.status, s.payment_channel,
    s.payable_at, s.expires_at, s.paid_at, jsonb_build_object('source', 'business_query_mock'),
    s.payable_at, COALESCE(s.paid_at, s.payable_at)
FROM seed_orders s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid
ON CONFLICT (order_uid) DO UPDATE SET
    status = EXCLUDED.status,
    expires_at = EXCLUDED.expires_at,
    paid_at = EXCLUDED.paid_at,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_payment_receipts (
    receipt_uid, payment_order_id, receipt_no, receipt_type, title, amount,
    status, issued_at, file_name, download_status, payload, created_at, updated_at
)
SELECT
    'PAY-RECEIPT-ARCHIVE-2026', p.id, 'DZPJ-20260613-ARCHIVE', 'electronic_invoice',
    '个人', p.amount, 'issued', '2026-06-13 15:45:00+08', '结果邮寄服务费电子票据.pdf',
    'available', jsonb_build_object('source', 'business_query_mock'),
    '2026-06-13 15:45:00+08', '2026-06-13 15:45:00+08'
FROM declaration_qa.business_payment_orders p
WHERE p.order_uid = 'PAY-REAL40-ARCHIVE-2026'
ON CONFLICT (receipt_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    updated_at = EXCLUDED.updated_at;

WITH seed_receipts (
    receipt_uid, apply_uid, receipt_no, receipt_type, issued_at, file_name
) AS (
    VALUES
        ('SUBMIT-RECEIPT-WATER-2026', 'AHZWFW-DOC40-MOCK-WATER-2026', 'TJHZ-20260710-WATER', 'submission', '2026-07-10 09:31:00+08'::timestamptz, '取水许可申请提交回执.pdf'),
        ('ACCEPT-RECEIPT-ARCHIVE-2026', 'AHZWFW-REAL40-MOCK-005', 'SLHZ-20260610-ARCHIVE', 'acceptance', '2026-06-10 10:06:00+08'::timestamptz, '档案材料收集受理回执.pdf')
)
INSERT INTO declaration_qa.business_submission_receipts (
    receipt_uid, apply_id, receipt_no, receipt_type, status, issued_at,
    file_name, download_status, payload, created_at, updated_at
)
SELECT
    s.receipt_uid, a.id, s.receipt_no, s.receipt_type, 'issued', s.issued_at,
    s.file_name, 'available', jsonb_build_object('source', 'business_query_mock'),
    s.issued_at, s.issued_at
FROM seed_receipts s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid
ON CONFLICT (receipt_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_result_certificates (
    certificate_uid, apply_id, certificate_name, certificate_no, certificate_type,
    status, issued_at, file_name, download_status, payload, created_at, updated_at
)
SELECT
    'RESULT-ARCHIVE-2026', a.id, '档案材料接收确认单', 'DAJS-2026-005',
    'electronic_result', 'issued', '2026-06-13 15:30:00+08',
    '档案材料接收确认单.pdf', 'available', jsonb_build_object('source', 'business_query_mock'),
    '2026-06-13 15:30:00+08', '2026-06-13 15:30:00+08'
FROM declaration_qa.business_applications a
WHERE a.apply_uid = 'AHZWFW-REAL40-MOCK-005'
ON CONFLICT (certificate_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_deliveries (
    delivery_uid, apply_id, delivery_type, status, courier_name, tracking_no,
    pickup_address, shipped_at, delivered_at, payload, created_at, updated_at
)
SELECT
    'DELIVERY-ARCHIVE-2026', a.id, 'mail', 'delivered', '中国邮政EMS',
    'EMS202606130005', '', '2026-06-13 17:00:00+08', '2026-06-15 14:10:00+08',
    jsonb_build_object('source', 'business_query_mock'),
    '2026-06-13 17:00:00+08', '2026-06-15 14:10:00+08'
FROM declaration_qa.business_applications a
WHERE a.apply_uid = 'AHZWFW-REAL40-MOCK-005'
ON CONFLICT (delivery_uid) DO UPDATE SET
    status = EXCLUDED.status,
    tracking_no = EXCLUDED.tracking_no,
    delivered_at = EXCLUDED.delivered_at,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_consultations (
    consultation_uid, user_id, consultation_type, subject, content_summary,
    status, channel, replied_at, reply_summary, payload, created_at, updated_at
) VALUES
    ('CONSULT-USER1-20260715-001', 1, 'consultation', '取水许可缴费期限咨询', '咨询缴费订单的支付期限和票据获取方式。', 'replied', 'online', '2026-07-15 11:20:00+08', '请在订单显示的支付期限内完成缴费，支付成功后可查询电子票据。', jsonb_build_object('source', 'business_query_mock'), '2026-07-15 10:00:00+08', '2026-07-15 11:20:00+08'),
    ('CONSULT-USER2-20260715-001', 2, 'complaint', '窗口等候时间投诉', '反映窗口等候时间较长。', 'processing', 'online', NULL, '', jsonb_build_object('source', 'business_query_mock'), '2026-07-15 09:00:00+08', '2026-07-15 09:00:00+08')
ON CONFLICT (consultation_uid) DO UPDATE SET
    status = EXCLUDED.status,
    replied_at = EXCLUDED.replied_at,
    reply_summary = EXCLUDED.reply_summary,
    updated_at = EXCLUDED.updated_at;

COMMIT;
