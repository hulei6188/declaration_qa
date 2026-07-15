-- Mock service data derived from published government-service documents.
-- Stable source_manifest_id values preserve the link back to curated sources.

BEGIN;

WITH seed_slots (
    slot_uid, source_manifest_id, item_code, item_name, region_code,
    hall_code, hall_name, hall_address, service_date, start_time, end_time,
    capacity, reserved_count, status
) AS (
    VALUES
        ('SLOT-DOC-WATER-20260720-0900', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-ZW-GCJS', '安徽省政务服务中心工程建设综合窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-20', TIME '09:00', TIME '10:00', 12, 5, 'available'),
        ('SLOT-DOC-WATER-20260720-1000', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-ZW-GCJS', '安徽省政务服务中心工程建设综合窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-20', TIME '10:00', TIME '11:00', 12, 12, 'full'),
        ('SLOT-DOC-WATER-20260720-1330', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-ZW-GCJS', '安徽省政务服务中心工程建设综合窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-20', TIME '13:30', TIME '14:30', 12, 2, 'available'),
        ('SLOT-DOC-GRASS-20260721-0900', 'item_30_official_ahzwfw', '11340000002985990J200016407900001', '草原防火期内在草原上进行爆破、勘察和施工等活动审批', '340000', 'AH-LYJ-ZW', '安徽省人民政府政务服务中心林业窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-21', TIME '09:00', TIME '10:00', 8, 3, 'available'),
        ('SLOT-DOC-GRASS-20260721-1400', 'item_30_official_ahzwfw', '11340000002985990J200016407900001', '草原防火期内在草原上进行爆破、勘察和施工等活动审批', '340000', 'AH-LYJ-ZW', '安徽省人民政府政务服务中心林业窗口', '安徽省合肥市包河区马鞍山路509号', DATE '2026-07-21', TIME '14:00', TIME '15:00', 8, 7, 'available'),
        ('SLOT-DOC-SPORT-20260722-0900', 'high_risk_sports_event_permit_official_ahzwfw', '11340000002985974W234013300500001', '举办高危险性体育赛事活动许可', '340000', 'AH-TYJ-01', '安徽省体育局政务服务窗口', '安徽省合肥市包河区芜湖路97号', DATE '2026-07-22', TIME '09:00', TIME '10:00', 10, 4, 'available'),
        ('SLOT-DOC-SPORT-20260722-1500', 'high_risk_sports_event_permit_official_ahzwfw', '11340000002985974W234013300500001', '举办高危险性体育赛事活动许可', '340000', 'AH-TYJ-01', '安徽省体育局政务服务窗口', '安徽省合肥市包河区芜湖路97号', DATE '2026-07-22', TIME '15:00', TIME '16:00', 10, 10, 'full'),
        ('SLOT-DOC-MARRIAGE-20260723-0800', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记', '340000', 'AH-MZT-102', '省涉外婚姻收养登记服务中心102室', '安徽省合肥市包河区屯溪路435号安徽省民政厅102室', DATE '2026-07-23', TIME '08:00', TIME '08:30', 6, 4, 'available'),
        ('SLOT-DOC-MARRIAGE-20260723-0900', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记', '340000', 'AH-MZT-102', '省涉外婚姻收养登记服务中心102室', '安徽省合肥市包河区屯溪路435号安徽省民政厅102室', DATE '2026-07-23', TIME '09:00', TIME '09:30', 6, 1, 'available'),
        ('SLOT-DOC-MARRIAGE-20260723-1430', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记', '340000', 'AH-MZT-102', '省涉外婚姻收养登记服务中心102室', '安徽省合肥市包河区屯溪路435号安徽省民政厅102室', DATE '2026-07-23', TIME '14:30', TIME '15:00', 6, 6, 'full'),
        ('SLOT-DOC-FIRE-20260724-0830', 'fire_accident_identification_review_official_ahzwfw', '11340000MB1C86468R234070901300001', '火灾事故认定复核', '340000', 'AH-XFJY-C1', '安徽省消防救援总队政务窗口', '安徽省合肥市广西路与中山路交叉口中山路C1-104号', DATE '2026-07-24', TIME '08:30', TIME '09:30', 8, 2, 'available'),
        ('SLOT-DOC-FIRE-20260724-1430', 'fire_accident_identification_review_official_ahzwfw', '11340000MB1C86468R234070901300001', '火灾事故认定复核', '340000', 'AH-XFJY-C1', '安徽省消防救援总队政务窗口', '安徽省合肥市广西路与中山路交叉口中山路C1-104号', DATE '2026-07-24', TIME '14:30', TIME '15:30', 8, 5, 'available')
)
INSERT INTO declaration_qa.business_appointment_slots (
    slot_uid, item_code, item_name, region_code, hall_code, hall_name,
    hall_address, service_date, start_time, end_time, capacity,
    reserved_count, status, payload
)
SELECT
    s.slot_uid, s.item_code, s.item_name, s.region_code, s.hall_code, s.hall_name,
    s.hall_address, s.service_date, s.start_time, s.end_time, s.capacity,
    s.reserved_count, s.status,
    jsonb_build_object(
        'source', 'submitted_document_mock',
        'source_manifest_id', s.source_manifest_id
    )
FROM seed_slots s
ON CONFLICT (slot_uid) DO UPDATE SET
    item_code = EXCLUDED.item_code,
    item_name = EXCLUDED.item_name,
    region_code = EXCLUDED.region_code,
    hall_code = EXCLUDED.hall_code,
    hall_name = EXCLUDED.hall_name,
    hall_address = EXCLUDED.hall_address,
    service_date = EXCLUDED.service_date,
    start_time = EXCLUDED.start_time,
    end_time = EXCLUDED.end_time,
    capacity = EXCLUDED.capacity,
    payload = EXCLUDED.payload,
    updated_at = CURRENT_TIMESTAMP;

WITH seed_appointments (
    appointment_uid, slot_uid, source_manifest_id, user_id, status,
    queue_number, people_ahead, estimated_wait_minutes, counter_name,
    checked_in_at, created_at, updated_at
) AS (
    VALUES
        ('APPT-USER1-WATER-20260720-0900', 'SLOT-DOC-WATER-20260720-0900', 'item_28_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 10:00:00+08', TIMESTAMPTZ '2026-07-15 10:00:00+08'),
        ('APPT-USER1-GRASS-20260721-1400', 'SLOT-DOC-GRASS-20260721-1400', 'item_30_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 10:10:00+08', TIMESTAMPTZ '2026-07-15 10:10:00+08'),
        ('APPT-USER1-SPORT-20260722-1500', 'SLOT-DOC-SPORT-20260722-1500', 'high_risk_sports_event_permit_official_ahzwfw', 1::bigint, 'cancelled', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-14 16:00:00+08', TIMESTAMPTZ '2026-07-15 09:20:00+08'),
        ('APPT-USER1-MARRIAGE-20260723-0900', 'SLOT-DOC-MARRIAGE-20260723-0900', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 11:00:00+08', TIMESTAMPTZ '2026-07-15 11:00:00+08'),
        ('APPT-USER1-FIRE-20260724-0830', 'SLOT-DOC-FIRE-20260724-0830', 'fire_accident_identification_review_official_ahzwfw', 1::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 11:15:00+08', TIMESTAMPTZ '2026-07-15 11:15:00+08'),
        ('APPT-USER2-MARRIAGE-20260723-0800', 'SLOT-DOC-MARRIAGE-20260723-0800', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', 2::bigint, 'booked', '', 0, 0, '', NULL::timestamptz, TIMESTAMPTZ '2026-07-15 12:00:00+08', TIMESTAMPTZ '2026-07-15 12:00:00+08')
)
INSERT INTO declaration_qa.business_appointments (
    appointment_uid, slot_id, user_id, item_code, item_name, region_code,
    hall_code, hall_name, hall_address, service_date, start_time, end_time,
    status, queue_number, people_ahead, estimated_wait_minutes, counter_name,
    checked_in_at, payload, created_at, updated_at
)
SELECT
    a.appointment_uid, s.id, a.user_id, s.item_code, s.item_name, s.region_code,
    s.hall_code, s.hall_name, s.hall_address, s.service_date, s.start_time, s.end_time,
    a.status, a.queue_number, a.people_ahead, a.estimated_wait_minutes, a.counter_name,
    a.checked_in_at,
    jsonb_build_object(
        'source', 'submitted_document_mock',
        'source_manifest_id', a.source_manifest_id
    ),
    a.created_at, a.updated_at
FROM seed_appointments a
JOIN declaration_qa.business_appointment_slots s ON s.slot_uid = a.slot_uid
ON CONFLICT (appointment_uid) DO UPDATE SET
    slot_id = EXCLUDED.slot_id,
    item_code = EXCLUDED.item_code,
    item_name = EXCLUDED.item_name,
    region_code = EXCLUDED.region_code,
    hall_code = EXCLUDED.hall_code,
    hall_name = EXCLUDED.hall_name,
    hall_address = EXCLUDED.hall_address,
    service_date = EXCLUDED.service_date,
    start_time = EXCLUDED.start_time,
    end_time = EXCLUDED.end_time,
    status = EXCLUDED.status,
    queue_number = EXCLUDED.queue_number,
    people_ahead = EXCLUDED.people_ahead,
    estimated_wait_minutes = EXCLUDED.estimated_wait_minutes,
    counter_name = EXCLUDED.counter_name,
    checked_in_at = EXCLUDED.checked_in_at,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

WITH seed_receipts (
    receipt_uid, apply_uid, source_manifest_id, receipt_no, receipt_type,
    issued_at, file_name
) AS (
    VALUES
        ('SUBMIT-RECEIPT-SPORT-2026', 'AHZWFW-REAL40-MOCK-007', 'high_risk_sports_event_permit_official_ahzwfw', 'TJHZ-20260706-SPORT', 'submission', TIMESTAMPTZ '2026-07-06 09:31:00+08', '高危险性体育赛事活动许可申请提交回执.pdf'),
        ('ACCEPT-RECEIPT-MARRIAGE-2026', 'AHZWFW-DOC40-MOCK-MARRIAGE-2026', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw', 'SLHZ-20260708-MARRIAGE', 'acceptance', TIMESTAMPTZ '2026-07-08 11:06:00+08', '涉外婚姻登记受理回执.pdf')
)
INSERT INTO declaration_qa.business_submission_receipts (
    receipt_uid, apply_id, receipt_no, receipt_type, status, issued_at,
    file_name, download_status, payload, created_at, updated_at
)
SELECT
    r.receipt_uid, a.id, r.receipt_no, r.receipt_type, 'issued', r.issued_at,
    r.file_name, 'available',
    jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', r.source_manifest_id),
    r.issued_at, r.issued_at
FROM seed_receipts r
JOIN declaration_qa.business_applications a ON a.apply_uid = r.apply_uid
ON CONFLICT (receipt_uid) DO UPDATE SET
    status = EXCLUDED.status,
    download_status = EXCLUDED.download_status,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

WITH seed_certificates (
    certificate_uid, apply_uid, source_manifest_id, certificate_name,
    certificate_no, issued_at, file_name
) AS (
    VALUES
        ('RESULT-ADOPTION-2025', 'AHZWFW-DOC40-MOCK-ADOPTION-2025', 'foreign_related_adoption_registration_official_ahzwfw', '涉外收养登记证', 'AH-SY-2025-0818', TIMESTAMPTZ '2025-08-18 15:10:00+08', '涉外收养登记证.pdf'),
        ('RESULT-OCS-2025', 'AHZWFW-DOC40-MOCK-OCS-2025', 'overseas_chinese_return_settlement_approval_official_ahzwfw', '华侨回国定居证', 'AH-HQDJ-2025-0320', TIMESTAMPTZ '2025-03-20 16:30:00+08', '华侨回国定居证.pdf')
)
INSERT INTO declaration_qa.business_result_certificates (
    certificate_uid, apply_id, certificate_name, certificate_no,
    certificate_type, status, issued_at, file_name, download_status,
    payload, created_at, updated_at
)
SELECT
    c.certificate_uid, a.id, c.certificate_name, c.certificate_no,
    'electronic_certificate', 'issued', c.issued_at, c.file_name, 'available',
    jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', c.source_manifest_id),
    c.issued_at, c.issued_at
FROM seed_certificates c
JOIN declaration_qa.business_applications a ON a.apply_uid = c.apply_uid
ON CONFLICT (certificate_uid) DO UPDATE SET
    status = EXCLUDED.status,
    file_name = EXCLUDED.file_name,
    download_status = EXCLUDED.download_status,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

WITH seed_deliveries (
    delivery_uid, apply_uid, source_manifest_id, delivery_type, status,
    courier_name, tracking_no, pickup_address, shipped_at, delivered_at,
    created_at, updated_at
) AS (
    VALUES
        ('DELIVERY-ADOPTION-2025', 'AHZWFW-DOC40-MOCK-ADOPTION-2025', 'foreign_related_adoption_registration_official_ahzwfw', 'self_pickup', 'ready_for_pickup', '', '', '安徽省合肥市包河区屯溪路435号安徽省民政厅123室', NULL::timestamptz, NULL::timestamptz, TIMESTAMPTZ '2025-08-18 15:15:00+08', TIMESTAMPTZ '2025-08-18 15:15:00+08'),
        ('DELIVERY-OCS-2025', 'AHZWFW-DOC40-MOCK-OCS-2025', 'overseas_chinese_return_settlement_approval_official_ahzwfw', 'mail', 'delivered', '中国邮政EMS', 'EMS202503200013', '', TIMESTAMPTZ '2025-03-20 17:00:00+08', TIMESTAMPTZ '2025-03-22 10:30:00+08', TIMESTAMPTZ '2025-03-20 17:00:00+08', TIMESTAMPTZ '2025-03-22 10:30:00+08')
)
INSERT INTO declaration_qa.business_deliveries (
    delivery_uid, apply_id, delivery_type, status, courier_name, tracking_no,
    pickup_address, shipped_at, delivered_at, payload, created_at, updated_at
)
SELECT
    d.delivery_uid, a.id, d.delivery_type, d.status, d.courier_name, d.tracking_no,
    d.pickup_address, d.shipped_at, d.delivered_at,
    jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', d.source_manifest_id),
    d.created_at, d.updated_at
FROM seed_deliveries d
JOIN declaration_qa.business_applications a ON a.apply_uid = d.apply_uid
ON CONFLICT (delivery_uid) DO UPDATE SET
    status = EXCLUDED.status,
    tracking_no = EXCLUDED.tracking_no,
    pickup_address = EXCLUDED.pickup_address,
    shipped_at = EXCLUDED.shipped_at,
    delivered_at = EXCLUDED.delivered_at,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

INSERT INTO declaration_qa.business_consultations (
    consultation_uid, user_id, consultation_type, subject, content_summary,
    status, channel, replied_at, reply_summary, payload, created_at, updated_at
) VALUES
    ('CONSULT-USER1-WATER-20260715', 1, 'consultation', '取水许可现场预约咨询', '咨询取水许可是否可以预约工程建设综合窗口。', 'replied', 'online', TIMESTAMPTZ '2026-07-15 09:20:00+08', '可以预约马鞍山路509号工程建设综合窗口，请按预约时段携带核验材料。', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'item_28_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 09:00:00+08', TIMESTAMPTZ '2026-07-15 09:20:00+08'),
    ('CONSULT-USER1-SPORT-20260715', 1, 'consultation', '高危险性体育赛事材料咨询', '咨询赛事活动许可的场地和安全保障材料。', 'processing', 'online', NULL, '', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'high_risk_sports_event_permit_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 10:30:00+08', TIMESTAMPTZ '2026-07-15 10:30:00+08'),
    ('CONSULT-USER1-MARRIAGE-20260715', 1, 'consultation', '涉外婚姻登记预约咨询', '咨询双方是否需要共同到场以及可预约时段。', 'replied', 'online', TIMESTAMPTZ '2026-07-15 11:40:00+08', '该事项为窗口办理，请双方按预约时间共同前往省涉外婚姻收养登记服务中心。', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'foreign_hmt_overseas_chinese_marriage_registration_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 11:10:00+08', TIMESTAMPTZ '2026-07-15 11:40:00+08'),
    ('CONSULT-USER2-FIRE-20260715', 2, 'consultation', '火灾事故认定复核办理地点咨询', '咨询火灾事故认定复核的线下办理地点。', 'replied', 'online', TIMESTAMPTZ '2026-07-15 14:20:00+08', '办理地点为中山路C1-104号安徽省消防救援总队政务窗口。', jsonb_build_object('source', 'submitted_document_mock', 'source_manifest_id', 'fire_accident_identification_review_official_ahzwfw'), TIMESTAMPTZ '2026-07-15 14:00:00+08', TIMESTAMPTZ '2026-07-15 14:20:00+08')
ON CONFLICT (consultation_uid) DO UPDATE SET
    status = EXCLUDED.status,
    replied_at = EXCLUDED.replied_at,
    reply_summary = EXCLUDED.reply_summary,
    payload = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;

COMMIT;
