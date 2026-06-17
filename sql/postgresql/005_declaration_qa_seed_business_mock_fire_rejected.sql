-- Virtual rejected business ticket for testing fire-prevention process routing.
--
-- This seed is intentionally scoped to apply_uid LIKE 'AHZWFW-MOCK-FIRE-%' so it
-- can be rerun without touching real business tickets.

BEGIN;

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-MOCK-FIRE-%'
)
DELETE FROM declaration_qa.business_reject_infos
WHERE apply_id IN (SELECT id FROM target_apps);

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-MOCK-FIRE-%'
)
DELETE FROM declaration_qa.business_application_statuses
WHERE apply_id IN (SELECT id FROM target_apps);

WITH seed (
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-MOCK-FIRE-001',
        1,
        'grassland_fire_blasting_approval',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-17 09:20:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'title', '防火办事流程',
            'item_name', '防火办事流程',
            'official_item_name', '草原防火期内在草原上进行爆破、勘察和施工等活动审批',
            'item_code', '11340000002985990J200016407900001',
            'source_file', 'item_30.md',
            'materials', jsonb_build_array('身份证件', '单位组织机构代码证', '申请表', '市县林业主管部门审查审核意见', '民用爆炸物品购买许可证', '爆破作业单位许可证'),
            'legal_limit', '20个工作日',
            'promise_limit', '10个工作日',
            'handle_form', '网上办理,移动端办理,窗口办理,快递申请',
            'consult_phone', '0551-62999817',
            'complaint_phone', '0551-62634463',
            'is_mock', true
        )
    )
)
INSERT INTO declaration_qa.business_applications (
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    payload,
    deleted_at,
    created_at,
    updated_at
)
SELECT
    apply_uid,
    user_id,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at::timestamptz,
    completed_at::timestamptz,
    payload,
    NULL::timestamptz,
    submitted_at::timestamptz,
    '2026-06-17 10:30:00+08'::timestamptz
FROM seed
ON CONFLICT (apply_uid) DO UPDATE SET
    user_id = EXCLUDED.user_id,
    biz_type = EXCLUDED.biz_type,
    region_code = EXCLUDED.region_code,
    status = EXCLUDED.status,
    current_node = EXCLUDED.current_node,
    node_name = EXCLUDED.node_name,
    submitted_at = EXCLUDED.submitted_at,
    completed_at = EXCLUDED.completed_at,
    payload = EXCLUDED.payload,
    deleted_at = NULL,
    updated_at = EXCLUDED.updated_at;

WITH seed_status (
    apply_uid,
    status,
    current_node,
    node_name,
    is_current,
    created_at,
    updated_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-MOCK-FIRE-001',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        TRUE,
        '2026-06-17 10:00:00+08'::timestamptz,
        '2026-06-17 10:30:00+08'::timestamptz,
        jsonb_build_object('step', '材料预审', 'message', '防火办事流程申请材料预审未通过，已退回补正。')
    )
)
INSERT INTO declaration_qa.business_application_statuses (
    apply_id,
    status,
    current_node,
    node_name,
    is_current,
    payload,
    created_at,
    updated_at
)
SELECT
    a.id,
    s.status,
    s.current_node,
    s.node_name,
    s.is_current,
    s.payload,
    s.created_at,
    s.updated_at
FROM seed_status s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid;

WITH seed_reject (
    apply_uid,
    material_code,
    material_name,
    reject_code,
    reject_reason,
    audit_node,
    is_current,
    created_at,
    updated_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-MOCK-FIRE-001',
        'FIRE_PROCESS_APPLICATION_FORM',
        '申请表',
        'FIRE_MEASURE_DESCRIPTION_INCOMPLETE',
        '申请表中的活动地点、作业时间和防火措施说明不完整，未能证明已采取必要防火措施。请补充具体作业地点、时间安排、防火责任人和现场防火措施后重新提交。',
        '材料预审',
        TRUE,
        '2026-06-17 10:10:00+08'::timestamptz,
        '2026-06-17 10:30:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'item_30.md',
            'official_item_name', '草原防火期内在草原上进行爆破、勘察和施工等活动审批',
            'accept_standard', '申请表应填写齐全，单位申请要加盖公章，内容真实详细准确。',
            'is_mock', true
        )
    )
)
INSERT INTO declaration_qa.business_reject_infos (
    apply_id,
    material_code,
    material_name,
    reject_code,
    reject_reason,
    audit_node,
    is_current,
    payload,
    created_at,
    updated_at
)
SELECT
    a.id,
    r.material_code,
    r.material_name,
    r.reject_code,
    r.reject_reason,
    r.audit_node,
    r.is_current,
    r.payload,
    r.created_at,
    r.updated_at
FROM seed_reject r
JOIN declaration_qa.business_applications a ON a.apply_uid = r.apply_uid;

COMMIT;
