-- Simulated tickets for the four currently published AHZWFW documents.
--
-- The seed targets demo user_id=1 and is idempotent. Rerunning it only
-- replaces applications whose apply_uid starts with AHZWFW-DOC40-MOCK-.

BEGIN;

DELETE FROM declaration_qa.business_applications
WHERE apply_uid LIKE 'AHZWFW-DOC40-MOCK-%';

WITH seed (
    apply_uid,
    user_id,
    item_code,
    item_name,
    biz_type,
    region_code,
    status,
    current_node,
    node_name,
    submitted_at,
    completed_at,
    updated_at,
    payload
) AS (
    VALUES
    (
        'AHZWFW-DOC40-MOCK-OCS-2025', 1,
        '11340000002986483M200014400100001', '华侨回国定居审批',
        'overseas_chinese_return_settlement_approval', '340000',
        'completed', 'FINISH', '办结',
        '2025-03-02 09:10:00+08'::timestamptz,
        '2025-03-20 16:30:00+08'::timestamptz,
        '2025-03-20 16:30:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'overseas_chinese_return_settlement_approval.md',
            'materials', jsonb_build_array('华侨回国定居申请表', '自愿放弃国外居留资格声明书', '中华人民共和国护照', '国外居留证明', '户口注销书面说明', '稳定生活保障证明', '房产证', '出入境记录'),
            'result', '准予华侨回国定居'
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-OCS-2026', 1,
        '11340000002986483M200014400100001', '华侨回国定居审批',
        'overseas_chinese_return_settlement_approval', '340000',
        'returned_for_correction', 'MATERIAL_REVIEW', '材料预审',
        '2026-07-01 10:05:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-10 14:20:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'overseas_chinese_return_settlement_approval.md',
            'materials', jsonb_build_array('华侨回国定居申请表', '自愿放弃国外居留资格声明书', '中华人民共和国护照', '国外居留证明', '户口注销书面说明', '稳定生活保障证明', '房产证', '出入境记录')
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-WATER-2026', 1,
        '11340000002985923L200011900100001', '取水许可',
        'water_withdrawal_permit', '340000',
        'in_review', 'TECHNICAL_REVIEW', '技术审查',
        '2026-07-03 09:25:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-11 11:40:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'item_28.md',
            'materials', jsonb_build_array('建设项目水资源论证报告书（表）', '水权转让可行性研究报告', '备案材料', '营业执照或身份证', '取水许可申请书', '利害关系说明')
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-ADOPTION-2025', 1,
        '11340000002986424J200071100200001', '涉外收养登记',
        'foreign_related_adoption_registration', '340000',
        'completed', 'FINISH', '办结',
        '2025-08-12 09:00:00+08'::timestamptz,
        '2025-08-18 15:10:00+08'::timestamptz,
        '2025-08-18 15:10:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'foreign_related_adoption_registration.md',
            'materials', jsonb_build_array('收养人护照', '来华收养子女通知书', '收养人照片', '送养人身份证', '被收养人照片', '授权委托书'),
            'result', '涉外收养登记完成'
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-ADOPTION-2026', 1,
        '11340000002986424J200071100200001', '涉外收养登记',
        'foreign_related_adoption_registration', '340000',
        'returned_for_correction', 'MATERIAL_REVIEW', '材料预审',
        '2026-07-05 14:15:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-09 16:45:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'foreign_related_adoption_registration.md',
            'materials', jsonb_build_array('收养人护照', '来华收养子女通知书', '收养人照片', '送养人身份证', '被收养人照片', '授权委托书')
        )
    ),
    (
        'AHZWFW-DOC40-MOCK-MARRIAGE-2026', 1,
        '11340000002986424J200071100100002', '涉外、涉港澳台、涉华侨结婚登记',
        'foreign_hmt_overseas_chinese_marriage_registration', '340000',
        'accepted', 'ACCEPT', '受理',
        '2026-07-08 10:30:00+08'::timestamptz,
        NULL::timestamptz,
        '2026-07-08 11:05:00+08'::timestamptz,
        jsonb_build_object(
            'source_file', 'foreign_hmt_overseas_chinese_marriage_registration.md',
            'materials', jsonb_build_array('内地居民身份证', '港澳台居民身份证', '港澳台居民通行证', '护照', '无血亲关系证明', '单身证明', '合照')
        )
    )
)
INSERT INTO declaration_qa.business_applications (
    apply_uid, user_id, item_code, item_name, biz_type, region_code,
    status, current_node, node_name, submitted_at, completed_at,
    payload, deleted_at, created_at, updated_at
)
SELECT
    apply_uid, user_id, item_code, item_name, biz_type, region_code,
    status, current_node, node_name, submitted_at, completed_at,
    payload || jsonb_build_object('item_code', item_code, 'item_name', item_name),
    NULL::timestamptz, submitted_at, updated_at
FROM seed;

WITH seed_status (
    apply_uid, sequence_no, from_status, status, current_node, node_name,
    reason_code, reason, material_code, material_name, is_current,
    occurred_at, source_event_id
) AS (
    VALUES
    ('AHZWFW-DOC40-MOCK-OCS-2025', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2025-03-02 09:10:00+08'::timestamptz, 'DOC40-OCS-2025-01'),
    ('AHZWFW-DOC40-MOCK-OCS-2025', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '', '', '', FALSE, '2025-03-04 10:20:00+08'::timestamptz, 'DOC40-OCS-2025-02'),
    ('AHZWFW-DOC40-MOCK-OCS-2025', 3, 'accepted', 'in_review', 'REVIEW', '审查', '', '', '', '', FALSE, '2025-03-10 09:30:00+08'::timestamptz, 'DOC40-OCS-2025-03'),
    ('AHZWFW-DOC40-MOCK-OCS-2025', 4, 'in_review', 'completed', 'FINISH', '办结', '', '审查通过，准予华侨回国定居。', '', '', TRUE, '2025-03-20 16:30:00+08'::timestamptz, 'DOC40-OCS-2025-04'),

    ('AHZWFW-DOC40-MOCK-OCS-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-01 10:05:00+08'::timestamptz, 'DOC40-OCS-2026-01'),
    ('AHZWFW-DOC40-MOCK-OCS-2026', 2, 'submitted', 'pre_accept', 'PRE_ACCEPT', '网上预审', '', '', '', '', FALSE, '2026-07-02 09:15:00+08'::timestamptz, 'DOC40-OCS-2026-02'),
    ('AHZWFW-DOC40-MOCK-OCS-2026', 3, 'pre_accept', 'returned_for_correction', 'MATERIAL_REVIEW', '材料预审', 'MISSING_PROPERTY_AND_TRAVEL_RECORD', '未提交房产证，且出入境记录缺少最近五年的完整记录，请补充后重新提交。', 'PROPERTY_AND_TRAVEL_RECORD', '房产证、出入境记录', TRUE, '2026-07-10 14:20:00+08'::timestamptz, 'DOC40-OCS-2026-03'),

    ('AHZWFW-DOC40-MOCK-WATER-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-03 09:25:00+08'::timestamptz, 'DOC40-WATER-2026-01'),
    ('AHZWFW-DOC40-MOCK-WATER-2026', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '', '', '', FALSE, '2026-07-04 14:00:00+08'::timestamptz, 'DOC40-WATER-2026-02'),
    ('AHZWFW-DOC40-MOCK-WATER-2026', 3, 'accepted', 'in_review', 'TECHNICAL_REVIEW', '技术审查', '', '正在审查建设项目水资源论证报告书及取水合理性。', '', '', TRUE, '2026-07-11 11:40:00+08'::timestamptz, 'DOC40-WATER-2026-03'),

    ('AHZWFW-DOC40-MOCK-ADOPTION-2025', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2025-08-12 09:00:00+08'::timestamptz, 'DOC40-ADOPTION-2025-01'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2025', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '', '', '', FALSE, '2025-08-13 10:00:00+08'::timestamptz, 'DOC40-ADOPTION-2025-02'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2025', 3, 'accepted', 'completed', 'FINISH', '办结', '', '材料审查通过，涉外收养登记已完成。', '', '', TRUE, '2025-08-18 15:10:00+08'::timestamptz, 'DOC40-ADOPTION-2025-03'),

    ('AHZWFW-DOC40-MOCK-ADOPTION-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-05 14:15:00+08'::timestamptz, 'DOC40-ADOPTION-2026-01'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2026', 2, 'submitted', 'pre_accept', 'PRE_ACCEPT', '网上预审', '', '', '', '', FALSE, '2026-07-06 09:40:00+08'::timestamptz, 'DOC40-ADOPTION-2026-02'),
    ('AHZWFW-DOC40-MOCK-ADOPTION-2026', 3, 'pre_accept', 'returned_for_correction', 'MATERIAL_REVIEW', '材料预审', 'AUTHORIZATION_UNSIGNED', '授权委托书缺少委托人签字，被收养人照片也未按要求提交，请补正后重新申报。', 'AUTHORIZATION_AND_PHOTO', '授权委托书、被收养人照片', TRUE, '2026-07-09 16:45:00+08'::timestamptz, 'DOC40-ADOPTION-2026-03'),

    ('AHZWFW-DOC40-MOCK-MARRIAGE-2026', 1, NULL, 'submitted', 'SUBMIT', '提交申请', '', '', '', '', FALSE, '2026-07-08 10:30:00+08'::timestamptz, 'DOC40-MARRIAGE-2026-01'),
    ('AHZWFW-DOC40-MOCK-MARRIAGE-2026', 2, 'submitted', 'accepted', 'ACCEPT', '受理', '', '申请材料已核验，等待登记办理。', '', '', TRUE, '2026-07-08 11:05:00+08'::timestamptz, 'DOC40-MARRIAGE-2026-02')
)
INSERT INTO declaration_qa.business_application_statuses (
    apply_id, sequence_no, from_status, status, current_node, node_name,
    reason_code, reason, material_code, material_name, source_event_id,
    is_current, payload, occurred_at, created_at, updated_at
)
SELECT
    a.id, s.sequence_no, s.from_status, s.status, s.current_node, s.node_name,
    s.reason_code, s.reason, s.material_code, s.material_name, s.source_event_id,
    s.is_current, jsonb_build_object('source', 'doc40_mock_seed'),
    s.occurred_at, s.occurred_at, s.occurred_at
FROM seed_status s
JOIN declaration_qa.business_applications a ON a.apply_uid = s.apply_uid
ORDER BY s.apply_uid, s.sequence_no;

WITH seed_reject (
    apply_uid, material_code, material_name, reject_code, reject_reason,
    audit_node, created_at
) AS (
    VALUES
    (
        'AHZWFW-DOC40-MOCK-OCS-2026',
        'PROPERTY_AND_TRAVEL_RECORD', '房产证、出入境记录',
        'MISSING_PROPERTY_AND_TRAVEL_RECORD',
        '未提交房产证，且出入境记录缺少最近五年的完整记录，请补充后重新提交。',
        '材料预审', '2026-07-10 14:20:00+08'::timestamptz
    ),
    (
        'AHZWFW-DOC40-MOCK-ADOPTION-2026',
        'AUTHORIZATION_AND_PHOTO', '授权委托书、被收养人照片',
        'AUTHORIZATION_UNSIGNED',
        '授权委托书缺少委托人签字，被收养人照片也未按要求提交，请补正后重新申报。',
        '材料预审', '2026-07-09 16:45:00+08'::timestamptz
    )
)
INSERT INTO declaration_qa.business_reject_infos (
    apply_id, material_code, material_name, reject_code, reject_reason,
    audit_node, is_current, payload, created_at, updated_at
)
SELECT
    a.id, r.material_code, r.material_name, r.reject_code, r.reject_reason,
    r.audit_node, TRUE, jsonb_build_object('source', 'doc40_mock_seed'),
    r.created_at, r.created_at
FROM seed_reject r
JOIN declaration_qa.business_applications a ON a.apply_uid = r.apply_uid;

COMMIT;
