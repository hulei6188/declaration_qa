-- 为现有演示数据库补充同一事项下的多个预约大厅和号源。

BEGIN;

WITH seed_slots (
    slot_uid, source_manifest_id, item_code, item_name, region_code,
    hall_code, hall_name, hall_address, service_date, start_time, end_time,
    capacity, reserved_count, status
) AS (
    VALUES
        ('SLOT-DOC-WATER-SLZW-20260720-0830', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-SLT-SLZW', '安徽省政务服务中心水利窗口', '安徽省合肥市包河区马鞍山路509号二楼水利服务区', DATE '2026-07-20', TIME '08:30', TIME '09:30', 10, 3, 'available'),
        ('SLOT-DOC-WATER-SLZW-20260720-1400', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-SLT-SLZW', '安徽省政务服务中心水利窗口', '安徽省合肥市包河区马鞍山路509号二楼水利服务区', DATE '2026-07-20', TIME '14:00', TIME '15:00', 10, 4, 'available'),
        ('SLOT-DOC-WATER-HFSW-20260720-1100', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-HF-SWZW', '合肥市政务服务中心水务窗口', '安徽省合肥市蜀山区怀宁路100号三楼水务服务区', DATE '2026-07-20', TIME '11:00', TIME '12:00', 8, 2, 'available'),
        ('SLOT-DOC-WATER-HFSW-20260720-1500', 'item_28_official_ahzwfw', '11340000002985923L200011900100001', '取水许可', '340000', 'AH-HF-SWZW', '合肥市政务服务中心水务窗口', '安徽省合肥市蜀山区怀宁路100号三楼水务服务区', DATE '2026-07-20', TIME '15:00', TIME '16:00', 8, 1, 'available')
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
        'source', 'multiple_halls_mock',
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
    reserved_count = EXCLUDED.reserved_count,
    status = EXCLUDED.status,
    payload = EXCLUDED.payload,
    updated_at = CURRENT_TIMESTAMP;

COMMIT;
