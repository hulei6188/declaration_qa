-- Simulated business ticket data generated from:
-- materials/curated/declaration_qa/2026-06-15-ahzwfw-real-10
--
-- This seed is intentionally scoped to apply_uid LIKE 'AHZWFW-REAL10-%' so it
-- can be rerun without touching real business tickets.

BEGIN;

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-REAL10-%'
)
DELETE FROM declaration_qa.business_reject_infos
WHERE apply_id IN (SELECT id FROM target_apps);

WITH target_apps AS (
    SELECT id
    FROM declaration_qa.business_applications
    WHERE apply_uid LIKE 'AHZWFW-REAL10-%'
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
        'AHZWFW-REAL10-001',
        1,
        'engineering_survey_design_master_selection',
        '340000',
        'in_review',
        'REVIEW',
        '审查',
        '2026-06-02 09:20:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '安徽省工程勘察设计大师评选',
            'item_code', '11340000MB0T3248X7234101703500001',
            'source_file', 'engineering_survey_design_master_selection.md',
            'materials', jsonb_build_array('申报材料意见函', '安徽省工程勘察设计大师申报表', '勘察设计大师申报附件材料'),
            'legal_limit', '20个工作日',
            'promise_limit', '20个工作日',
            'handle_form', '网上办理,窗口办理,快递申请'
        )
    ),
    (
        'AHZWFW-REAL10-002',
        1,
        'second_level_constructor_additional_registration',
        '340000',
        'accepted',
        'ACCEPT',
        '受理',
        '2026-06-03 10:10:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级建造师增项注册',
            'item_code', '11340000MB0T3248X7200011705500202',
            'source_file', 'second_level_constructor_additional_registration.md',
            'materials', jsonb_build_array('二级建造师增项注册表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理,自助端办理'
        )
    ),
    (
        'AHZWFW-REAL10-003',
        1,
        'second_level_constructor_initial_registration',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-04 14:35:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级建造师初始注册',
            'item_code', '11340000MB0T3248X7200011705500205',
            'source_file', 'second_level_constructor_initial_registration.md',
            'materials', jsonb_build_array('二级建造师初始注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-004',
        1,
        'second_level_constructor_re_registration',
        '340000',
        'in_review',
        'REVIEW',
        '审查',
        '2026-06-05 09:05:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级建造师重新注册',
            'item_code', '11340000MB0T3248X7200011705500209',
            'source_file', 'second_level_constructor_re_registration.md',
            'materials', jsonb_build_array('二级建造师重新注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-005',
        1,
        'second_level_constructor_renewal_registration',
        '340000',
        'completed',
        'FINISH',
        '办结',
        '2026-05-28 11:25:00+08',
        '2026-06-06 16:10:00+08',
        jsonb_build_object(
            'item_name', '二级建造师延续注册',
            'item_code', '11340000MB0T3248X7200011705500206',
            'source_file', 'second_level_constructor_renewal_registration.md',
            'materials', jsonb_build_array('二级建造师延续注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-006',
        1,
        'second_level_cost_engineer_cancellation_registration',
        '340000',
        'pre_accept',
        'PRE_ACCEPT',
        '网上预审',
        '2026-06-07 15:40:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师注销注册',
            'item_code', '11340000MB0T3248X7200011705600003',
            'source_file', 'second_level_cost_engineer_cancellation_registration.md',
            'materials', jsonb_build_array('二级造价工程师注销注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-007',
        1,
        'second_level_cost_engineer_change_registration',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-08 09:55:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师变更注册',
            'item_code', '11340000MB0T3248X7200011705600005',
            'source_file', 'second_level_cost_engineer_change_registration.md',
            'materials', jsonb_build_array('二级造价工程师变更注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-008',
        1,
        'second_level_cost_engineer_initial_registration',
        '340000',
        'in_decision',
        'DECISION',
        '决定',
        '2026-06-09 13:50:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师初始注册',
            'item_code', '11340000MB0T3248X7200011705600002',
            'source_file', 'second_level_cost_engineer_initial_registration.md',
            'materials', jsonb_build_array('二级造价工程师初始注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-009',
        1,
        'second_level_cost_engineer_renewal_registration',
        '340000',
        'rejected',
        'MATERIAL_REVIEW',
        '材料预审',
        '2026-06-10 10:05:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级造价工程师延续注册',
            'item_code', '11340000MB0T3248X7200011705600004',
            'source_file', 'second_level_cost_engineer_renewal_registration.md',
            'materials', jsonb_build_array('二级造价工程师延续注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
        )
    ),
    (
        'AHZWFW-REAL10-010',
        1,
        'second_level_structural_engineer_initial_registration',
        '340000',
        'accepted',
        'ACCEPT',
        '受理',
        '2026-06-11 16:25:00+08',
        NULL::timestamptz,
        jsonb_build_object(
            'item_name', '二级结构工程师初始注册',
            'item_code', '11340000MB0T3248X7200011703900206',
            'source_file', 'second_level_structural_engineer_initial_registration.md',
            'materials', jsonb_build_array('二级注册结构工程师初始注册申请表', '承诺函'),
            'legal_limit', '20个工作日',
            'promise_limit', '8个工作日',
            'handle_form', '网上办理,窗口办理'
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
    COALESCE(completed_at::timestamptz, submitted_at::timestamptz + INTERVAL '2 days')
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
    ('AHZWFW-REAL10-001', 'in_review', 'REVIEW', '审查', TRUE, '2026-06-04 09:00:00+08'::timestamptz, '2026-06-12 10:20:00+08'::timestamptz, jsonb_build_object('step', '审查', 'message', '已进入主管部门审查环节，正在核验申报材料和业绩证明。')),
    ('AHZWFW-REAL10-002', 'accepted', 'ACCEPT', '受理', TRUE, '2026-06-03 11:30:00+08'::timestamptz, '2026-06-12 09:10:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '窗口已受理二级建造师增项注册申请，等待进入审查环节。')),
    ('AHZWFW-REAL10-003', 'rejected', 'MATERIAL_REVIEW', '材料预审', TRUE, '2026-06-05 10:15:00+08'::timestamptz, '2026-06-12 14:00:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '材料预审未通过，申请表信息需补正后重新提交。')),
    ('AHZWFW-REAL10-004', 'in_review', 'REVIEW', '审查', TRUE, '2026-06-06 09:20:00+08'::timestamptz, '2026-06-13 10:40:00+08'::timestamptz, jsonb_build_object('step', '审查', 'message', '二级建造师重新注册申请正在审查。')),
    ('AHZWFW-REAL10-005', 'completed', 'FINISH', '办结', TRUE, '2026-06-06 16:10:00+08'::timestamptz, '2026-06-06 16:10:00+08'::timestamptz, jsonb_build_object('step', '办结', 'message', '二级建造师延续注册已办结，结果已反馈。')),
    ('AHZWFW-REAL10-006', 'pre_accept', 'PRE_ACCEPT', '网上预审', TRUE, '2026-06-08 09:00:00+08'::timestamptz, '2026-06-13 15:30:00+08'::timestamptz, jsonb_build_object('step', '网上预审', 'message', '二级造价工程师注销注册正在网上预审。')),
    ('AHZWFW-REAL10-007', 'rejected', 'MATERIAL_REVIEW', '材料预审', TRUE, '2026-06-09 10:30:00+08'::timestamptz, '2026-06-13 16:05:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '承诺函未完成在线确认，已退回补正。')),
    ('AHZWFW-REAL10-008', 'in_decision', 'DECISION', '决定', TRUE, '2026-06-11 09:20:00+08'::timestamptz, '2026-06-14 11:50:00+08'::timestamptz, jsonb_build_object('step', '决定', 'message', '二级造价工程师初始注册已通过审查，等待作出核准决定。')),
    ('AHZWFW-REAL10-009', 'rejected', 'MATERIAL_REVIEW', '材料预审', TRUE, '2026-06-11 14:25:00+08'::timestamptz, '2026-06-14 15:40:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '延续注册申请表不符合受理标准，已退回补正。')),
    ('AHZWFW-REAL10-010', 'accepted', 'ACCEPT', '受理', TRUE, '2026-06-12 09:35:00+08'::timestamptz, '2026-06-14 17:10:00+08'::timestamptz, jsonb_build_object('step', '受理', 'message', '二级结构工程师初始注册申请已受理。'))
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
        'AHZWFW-REAL10-003',
        'CONSTRUCTOR_INITIAL_APPLICATION_FORM',
        '二级建造师初始注册申请表',
        'FORM_CONTENT_INCONSISTENT',
        '申请表中注册专业和聘用单位信息填写不一致。请按示例样表核对后重新提交，表格内容需真实无误。',
        '材料预审',
        TRUE,
        '2026-06-12 13:30:00+08'::timestamptz,
        '2026-06-12 14:00:00+08'::timestamptz,
        jsonb_build_object('source_file', 'second_level_constructor_initial_registration.md', 'accept_standard', '表格内容填写真实无误。')
    ),
    (
        'AHZWFW-REAL10-007',
        'COMMITMENT_LETTER',
        '承诺函',
        'COMMITMENT_NOT_CONFIRMED',
        '承诺函未由企业或个人作为责任主体完成在线勾选确认。请重新进入申报系统完成承诺后提交。',
        '材料预审',
        TRUE,
        '2026-06-13 15:30:00+08'::timestamptz,
        '2026-06-13 16:05:00+08'::timestamptz,
        jsonb_build_object('source_file', 'second_level_cost_engineer_change_registration.md', 'accept_standard', '本承诺函需由企业或个人作为责任主体作出承诺，申报时通过在线勾选完成提交。')
    ),
    (
        'AHZWFW-REAL10-009',
        'COST_RENEWAL_APPLICATION_FORM',
        '二级造价工程师延续注册申请表',
        'CONTINUING_EDUCATION_NOT_CONFIRMED',
        '延续注册申请表未体现注册有效期届满30日内通过注册专业继续教育的相关信息，请补充后重新提交。',
        '材料预审',
        TRUE,
        '2026-06-14 15:05:00+08'::timestamptz,
        '2026-06-14 15:40:00+08'::timestamptz,
        jsonb_build_object('source_file', 'second_level_cost_engineer_renewal_registration.md', 'accept_standard', '在注册有效期届满30日内，通过注册专业继续教育后申请延续注册。')
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
