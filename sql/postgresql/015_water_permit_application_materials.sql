-- Publish the water-permit v2 schema with mandatory application-material uploads.

BEGIN;

UPDATE declaration_qa.business_item_form_schemas
SET status = 'deprecated',
    updated_at = CURRENT_TIMESTAMP
WHERE item_code = '11340000002985923L200011900100001'
  AND region_code = ''
  AND version_no < 2;

INSERT INTO declaration_qa.business_item_form_schemas (
    schema_uid, item_code, item_name, biz_type, region_code, version_no, status, fields, metadata
) VALUES (
    'water-permit-v2',
    '11340000002985923L200011900100001',
    '取水许可',
    'water_permit',
    '',
    2,
    'published',
    '[
      {"name":"applicant_name","label":"申请单位或个人","type":"text","required":true,"max_length":200,"aliases":["申请人","单位名称"]},
      {"name":"credit_code","label":"统一社会信用代码或证件号码","type":"text","required":true,"max_length":40,"aliases":["统一社会信用代码","证件号码"]},
      {"name":"contact_name","label":"联系人","type":"text","required":true,"max_length":80,"aliases":["经办人"]},
      {"name":"contact_phone","label":"联系电话","type":"phone","required":true,"aliases":["手机号","电话"]},
      {"name":"water_source","label":"取水水源","type":"text","required":true,"max_length":200,"aliases":["水源"]},
      {"name":"annual_water_volume","label":"年取水量（立方米）","type":"number","required":true,"aliases":["年取水量","取水量"]},
      {"name":"water_use","label":"取水用途","type":"multiselect","required":true,"options":["制水供水","原水供水","生活用水","工业用水","农业用水","其他"],"aliases":["用途"]},
      {
        "name":"application_materials",
        "label":"申报材料",
        "type":"material_group",
        "required":true,
        "description":"支持 PDF、Word、Excel、JPG、PNG、ZIP；单个文件不超过 20 MB。",
        "accept":[".pdf",".doc",".docx",".xls",".xlsx",".jpg",".jpeg",".png",".zip"],
        "materials":[
          {"key":"water_resource_report","label":"建设项目水资源论证报告书（表）","required":true,"max_files":3},
          {"key":"water_right_transfer_report","label":"水权转让可行性研究报告","required":false,"description":"通过水权转让方式获得取水指标时上传","max_files":3},
          {"key":"filing_materials","label":"备案材料","required":false,"description":"属于备案项目时上传","max_files":3},
          {"key":"business_license_or_id","label":"营业执照或身份证","required":true,"max_files":2},
          {"key":"water_permit_application","label":"取水许可申请书","required":true,"max_files":2},
          {"key":"interested_party_statement","label":"利害关系说明","required":true,"max_files":3}
        ]
      }
    ]'::jsonb,
    jsonb_build_object(
        'source', 'water_permit_application_materials',
        'source_item_code', '11340000002985923L200011900100001'
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

