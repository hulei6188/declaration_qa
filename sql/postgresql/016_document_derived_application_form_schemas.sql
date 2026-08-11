-- Publish application form schemas derived from uploaded, published service guides.

BEGIN;

INSERT INTO declaration_qa.business_item_form_schemas (
    schema_uid, item_code, item_name, biz_type, region_code, version_no, status, fields, metadata
) VALUES
(
    'disabled-veterans-centralized-support-v1',
    '11340000MB1527779M200072400100001',
    '对退出现役残疾军人集中供养的确定',
    'disabled_veterans_centralized_support_determination',
    '340000000000',
    1,
    'published',
    '[
      {"name":"applicant_name","label":"申请人姓名","type":"text","required":true,"max_length":80,"aliases":["姓名","残疾军人姓名"]},
      {"name":"id_number","label":"身份证号码","type":"id_number","required":true,"aliases":["证件号码","身份证号"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},
      {"name":"household_address","label":"户籍地址","type":"text","required":true,"max_length":300,"aliases":["户籍所在地","户口所在地"]},
      {"name":"disability_level","label":"残疾等级","type":"select","required":true,"options":["一级","二级","三级","四级"],"aliases":["伤残等级","残疾军人等级"]},
      {"name":"support_reason","label":"集中供养申请原因","type":"multiselect","required":true,"options":["需要长年医疗","独身一人不便分散安置"],"aliases":["申请原因","供养原因"]},
      {
        "name":"application_materials",
        "label":"申报材料",
        "type":"material_group",
        "required":true,
        "description":"办事指南列明材料形式为纸质；线上提交扫描件用于预审，窗口办理时按要求核验纸质材料。",
        "accept":[".pdf",".jpg",".jpeg",".png"],
        "materials":[
          {"key":"resident_id_card","label":"中华人民共和国居民身份证","required":true,"max_files":2},
          {"key":"disabled_veteran_certificate","label":"残疾军人证","required":true,"max_files":2},
          {"key":"household_register","label":"中华人民共和国居民户口簿","required":true,"max_files":5},
          {"key":"written_application","label":"个人书面申请","required":true,"max_files":2}
        ]
      }
    ]'::jsonb,
    jsonb_build_object(
        'source', 'published_document',
        'source_document_id', 22,
        'source_document_uid', 'doc-86270ebb951000cb259955b5e4a0f2c9',
        'policy_version', 'ahzwfw-2026-06-15',
        'source_material_count', 4
    )
),
(
    'fire-accident-review-v1',
    '11340000MB1C86468R234070901300001',
    '火灾事故认定复核',
    'fire_accident_review',
    '340000000000',
    1,
    'published',
    '[
      {"name":"applicant_type","label":"申请人类型","type":"select","required":true,"options":["自然人","法人"],"aliases":["主体类型","申请主体"]},
      {"name":"applicant_name","label":"申请人名称","type":"text","required":true,"max_length":200,"aliases":["申请人姓名","单位名称"]},
      {"name":"id_number_or_credit_code","label":"证件号码或统一社会信用代码","type":"text","required":true,"max_length":40,"aliases":["身份证号","证件号码","统一社会信用代码"]},
      {"name":"contact_name","label":"联系人","type":"text","required":true,"max_length":80,"aliases":["经办人"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},
      {"name":"respondent_name","label":"被申请人名称","type":"text","required":true,"max_length":200,"aliases":["原认定机构","被申请人"]},
      {"name":"determination_number","label":"火灾事故认定书编号","type":"text","required":true,"max_length":100,"aliases":["认定书编号","事故认定编号"]},
      {"name":"determination_served_date","label":"认定书送达日期","type":"date","required":true,"aliases":["送达日期","收到认定书日期"]},
      {"name":"review_request","label":"复核请求","type":"textarea","required":true,"min_length":5,"max_length":1000,"aliases":["复核诉求","申请请求"]},
      {"name":"review_facts_reasons_evidence","label":"主要事实、理由和证据","type":"textarea","required":true,"min_length":10,"max_length":4000,"aliases":["事实理由","复核理由","证据"]},
      {"name":"application_date","label":"申请复核日期","type":"date","required":true,"aliases":["申请日期"]}
    ]'::jsonb,
    jsonb_build_object(
        'source', 'published_document',
        'source_document_id', 28,
        'source_document_uid', 'doc-25ecd3d977144b175dd78b4203b8d367',
        'policy_version', 'ahzwfw-2026-06-15',
        'source_material_count', 0,
        'note', '申请材料章节标注无；表单字段依据受理条件中的书面复核申请要素设置'
    )
),
(
    'foreign-marriage-registration-v1',
    '11340000002986424J200071100100002',
    '涉外、涉港澳台、涉华侨结婚登记',
    'foreign_hmt_overseas_chinese_marriage_registration',
    '340000000000',
    1,
    'published',
    '[
      {"name":"mainland_party_name","label":"内地居民姓名","type":"text","required":true,"max_length":80,"aliases":["内地一方姓名","内地居民"]},
      {"name":"mainland_party_id_number","label":"内地居民身份证号码","type":"id_number","required":true,"aliases":["内地一方身份证号","居民身份证号"]},
      {"name":"other_party_name","label":"另一方姓名","type":"text","required":true,"max_length":120,"aliases":["外籍一方姓名","港澳台一方姓名","华侨姓名"]},
      {"name":"other_party_identity_type","label":"另一方身份类型","type":"select","required":true,"options":["外国人","香港居民","澳门居民","台湾居民","华侨"],"aliases":["身份类别","人员类型"]},
      {"name":"other_party_document_number","label":"另一方证件号码","type":"id_number","required":true,"aliases":["护照号码","通行证号码","港澳台身份证号"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},
      {
        "name":"application_materials",
        "label":"申报材料",
        "type":"material_group",
        "required":true,
        "description":"身份材料按另一方身份类型选择提交；指南列明材料形式为纸质，登记时按要求核验原件。",
        "accept":[".pdf",".jpg",".jpeg",".png"],
        "materials":[
          {"key":"mainland_resident_id_card","label":"内地居民身份证","required":true,"max_files":2},
          {"key":"hmt_resident_id_card","label":"港澳台居民身份证","required":false,"description":"另一方为香港、澳门或台湾居民时提交","max_files":2},
          {"key":"hmt_travel_permit","label":"港澳台居民通行证","required":false,"description":"另一方为香港、澳门或台湾居民时提交","max_files":2},
          {"key":"passport","label":"护照","required":false,"description":"另一方为外国人或华侨时提交","max_files":2},
          {"key":"no_lineal_or_collateral_blood_relationship_proof","label":"无直系血亲和三代以内旁系血亲关系证明","required":true,"max_files":2},
          {"key":"single_status_proof","label":"无配偶证明","required":true,"max_files":2},
          {"key":"joint_photo","label":"双方合照","required":true,"max_files":3}
        ]
      }
    ]'::jsonb,
    jsonb_build_object(
        'source', 'published_document',
        'source_document_id', 30,
        'source_document_uid', 'doc-e3b1bfe2394c94cc52c711cf89eeee66',
        'policy_version', 'ahzwfw-2026-06-15',
        'source_material_count', 7
    )
),
(
    'foreign-adoption-registration-v1',
    '11340000002986424J200071100200001',
    '涉外收养登记',
    'foreign_related_adoption_registration',
    '340000000000',
    1,
    'published',
    '[
      {"name":"adopter_name","label":"收养人姓名","type":"text","required":true,"max_length":120,"aliases":["申请收养人","收养申请人"]},
      {"name":"adopter_passport_number","label":"收养人护照号码","type":"id_number","required":true,"aliases":["护照号","护照号码"]},
      {"name":"adopter_nationality","label":"收养人国籍","type":"text","required":true,"max_length":80,"aliases":["国籍"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},
      {"name":"adoptee_name","label":"被收养人姓名","type":"text","required":true,"max_length":80,"aliases":["被收养儿童姓名","被收养人"]},
      {"name":"china_adoption_notice_number","label":"来华收养子女通知书编号","type":"text","required":true,"max_length":120,"aliases":["来华收养通知书编号","通知书编号"]},
      {
        "name":"application_materials",
        "label":"申报材料",
        "type":"material_group",
        "required":true,
        "description":"指南支持纸质或电子材料；夫妻共同收养且一方无法到场时需提交经公证、认证的授权委托书。",
        "accept":[".pdf",".jpg",".jpeg",".png"],
        "materials":[
          {"key":"adopter_passport","label":"收养人护照","required":true,"max_files":4},
          {"key":"china_adoption_notice","label":"来华收养子女通知书","required":true,"max_files":2},
          {"key":"adopter_photo","label":"收养人照片","required":true,"max_files":4},
          {"key":"placing_party_id_card","label":"送养人身份证","required":true,"max_files":2},
          {"key":"adoptee_photo","label":"被收养人照片","required":true,"max_files":4},
          {"key":"power_of_attorney","label":"授权委托书","required":false,"description":"夫妻共同收养且一方无法到场时提交","max_files":2}
        ]
      }
    ]'::jsonb,
    jsonb_build_object(
        'source', 'published_document',
        'source_document_id', 32,
        'source_document_uid', 'doc-15b17087cb2c978dff5c18ccee1608ca',
        'policy_version', 'ahzwfw-2026-06-15',
        'source_material_count', 6
    )
),
(
    'high-risk-sports-event-permit-v1',
    '11340000002985974W234013300500001',
    '举办高危险性体育赛事活动许可',
    'high_risk_sports_event_permit',
    '340000000000',
    1,
    'published',
    '[
      {"name":"organizer_name","label":"赛事活动组织者名称","type":"text","required":true,"max_length":200,"aliases":["申请人","主办方","组织者"]},
      {"name":"credit_code_or_id_number","label":"统一社会信用代码或证件号码","type":"text","required":true,"max_length":40,"aliases":["统一社会信用代码","身份证号","证件号码"]},
      {"name":"contact_name","label":"联系人","type":"text","required":true,"max_length":80,"aliases":["经办人"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话","联系方式"]},
      {"name":"event_name","label":"赛事活动名称","type":"text","required":true,"max_length":200,"aliases":["赛事名称","活动名称"]},
      {"name":"event_type","label":"赛事类别","type":"select","required":true,"options":["潜水赛事","航空运动","登山","攀岩或攀冰","滑雪登山","汽车摩托车赛事"],"aliases":["赛事类型","活动类别"]},
      {"name":"event_start_date","label":"赛事开始日期","type":"date","required":true,"aliases":["开始时间","举办日期"]},
      {"name":"event_end_date","label":"赛事结束日期","type":"date","required":true,"aliases":["结束时间"]},
      {"name":"event_location","label":"赛事举办地点","type":"text","required":true,"max_length":300,"aliases":["举办地点","场地"]},
      {"name":"participant_scale","label":"参赛规模（人）","type":"number","required":true,"aliases":["参赛人数","赛事规模"]},
      {"name":"co_organizers","label":"承办方、协办方","type":"textarea","required":false,"max_length":1000,"aliases":["承办单位","协办单位"]},
      {"name":"participation_conditions","label":"参赛条件","type":"textarea","required":true,"max_length":2000,"aliases":["报名条件","参赛资格"]},
      {
        "name":"application_materials",
        "label":"申报材料",
        "type":"material_group",
        "required":true,
        "description":"指南列明前五项材料为必要且可容缺后补；空域使用审批材料仅航空运动相关赛事提交。",
        "accept":[".pdf",".doc",".docx",".xls",".xlsx",".jpg",".jpeg",".png",".zip"],
        "materials":[
          {"key":"event_application","label":"举办高危险性体育赛事活动申请书","required":true,"max_files":2},
          {"key":"professional_qualification_proof","label":"专业技术人员的资格或资质证明材料","required":true,"max_files":10},
          {"key":"venue_equipment_facility_compliance","label":"场地、器材和设施符合相关标准和要求的说明性材料","required":true,"max_files":10},
          {"key":"organizer_responsibility_agreement","label":"赛事活动组织者权利义务和责任分工书面协议","required":true,"max_files":5},
          {"key":"risk_and_safety_plans","label":"风险评估、风险防范、应急处置、安全工作、熔断机制和赛事组织方案等材料","required":true,"max_files":10},
          {"key":"airspace_use_approval","label":"空域使用审批材料","required":false,"description":"航空运动相关赛事提交","max_files":5}
        ]
      }
    ]'::jsonb,
    jsonb_build_object(
        'source', 'published_document',
        'source_document_id', 36,
        'source_document_uid', 'doc-7ee0a86b5d76199629df81ad9e2c0508',
        'policy_version', 'ahzwfw-2026-06-15',
        'source_material_count', 6
    )
)
ON CONFLICT (item_code, region_code, version_no) DO UPDATE SET
    schema_uid = EXCLUDED.schema_uid,
    item_name = EXCLUDED.item_name,
    biz_type = EXCLUDED.biz_type,
    status = EXCLUDED.status,
    fields = EXCLUDED.fields,
    metadata = EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

COMMIT;
