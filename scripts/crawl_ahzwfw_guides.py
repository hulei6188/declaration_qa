"""从安徽政务服务网抓取材料丰富且未重复的省级个人办事指南。"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from sqlalchemy import create_engine, text

from gov_service_agent.core.config import get_settings


BASE_URL = "https://www.ahzwfw.gov.cn"
SOURCE_PAGE = f"{BASE_URL}/bog-bsdt/static/workProcess.html?subjecttype=P"
THEME_API = f"{BASE_URL}/bog-bsdt/commonConfig/serviceObjects.do"
LIST_API = f"{BASE_URL}/bog-bsdt/implement/queryImplements.do"
DETAIL_API = f"{BASE_URL}/bog-bsdt/implement/queryImplementInfoNew.do"
EVALUATE_API = f"{BASE_URL}/bog-bsdt/portal/getEvaluatePercent.do"
DETAIL_PAGE = f"{BASE_URL}/bog-bsdt/static/workProcess/components/applicationMaterial.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) declaration-qa/1.0",
    "Referer": SOURCE_PAGE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        default="materials/curated/declaration_qa/2026-07-20-ahzwfw-diverse-rich-20",
    )
    return parser.parse_args()


def request_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    *,
    attempts: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.post(url, data=payload, timeout=45)
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                return body
            raise ValueError("response is not a JSON object")
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last_error}")


def session() -> requests.Session:
    value = requests.Session()
    value.headers.update(HEADERS)
    return value


def json_value(value: Any, default: Any) -> Any:
    if isinstance(value, type(default)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return default
        return parsed if isinstance(parsed, type(default)) else default
    return default


def plain_text(value: Any) -> str:
    result = html.unescape(str(value or ""))
    result = re.sub(r"<br\s*/?>", "\n", result, flags=re.IGNORECASE)
    result = re.sub(r"</(?:p|div|li|tr|h[1-6])>", "\n", result, flags=re.IGNORECASE)
    result = re.sub(r"<[^>]+>", "", result)
    result = result.replace("\r\n", "\n").replace("\r", "\n")
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def cell(value: Any) -> str:
    result = plain_text(value) or "无"
    return result.replace("|", "\\|").replace("\n", "<br>")


def normalize_name(value: Any) -> str:
    result = plain_text(value)
    result = re.sub(r"办事指南$", "", result)
    result = re.sub(r"[\s（）()·,，。;；:：/\\—_-]+", "", result)
    return result.lower()


def collect_existing(root: Path) -> tuple[set[str], set[str], set[str]]:
    ssqd_codes: set[str] = set()
    item_codes: set[str] = set()
    names: set[str] = set()
    for manifest_path in root.glob("materials/**/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for document in manifest.get("documents") or []:
            if not isinstance(document, dict):
                continue
            ssqd_codes.add(str(document.get("ssqdCode") or document.get("ssqd_code") or "").strip())
            item_codes.add(str(document.get("item_code") or document.get("ssqdBaseCode") or "").strip())
            names.add(normalize_name(document.get("item_name") or document.get("title")))

    try:
        engine = create_engine(get_settings().effective_database_url)
        with engine.begin() as connection:
            rows = connection.execute(
                text("SELECT title, metadata FROM declaration_qa.documents")
            ).mappings()
            for row in rows:
                metadata = dict(row.get("metadata") or {})
                ssqd_codes.add(str(metadata.get("ssqd_code") or "").strip())
                item_codes.add(str(metadata.get("item_code") or "").strip())
                names.add(normalize_name(metadata.get("item_name") or row.get("title")))
    except Exception as exc:  # 数据库不可用时仍可按材料清单去重
        print(f"warning: database dedupe unavailable: {type(exc).__name__}: {exc}", flush=True)

    ssqd_codes.discard("")
    item_codes.discard("")
    names.discard("")
    return ssqd_codes, item_codes, names


def fetch_themes(client: requests.Session) -> tuple[dict[str, str], dict[str, Any]]:
    body = request_json(client, THEME_API, {"serviceObject": "P"})
    themes = body.get("data") or []
    mapping = {
        str(item.get("dm") or ""): plain_text(item.get("mc"))
        for item in themes
        if isinstance(item, dict)
    }
    return mapping, body


def fetch_list(client: requests.Session) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = {
            "currentPageNo": page,
            "pageSize": 50,
            "ssqdZoneCode": "340000000000",
            "ztmb": "",
            "ssqdTypeCode": "",
            "ssqdName": "",
            "isOnline": "",
            "allOnline": "",
            "iscsjtb": "",
            "serviceObject": "P",
            "xzqhZj": "0",
            "isPage": "1",
        }
        body = request_json(client, LIST_API, payload)
        responses.append(body)
        data = body.get("data") or {}
        page_rows = [item for item in data.get("rows") or [] if isinstance(item, dict)]
        rows.extend(page_rows)
        total = int(data.get("total") or 0)
        print(f"list page {page}: {len(page_rows)} rows ({len(rows)}/{total})", flush=True)
        if not page_rows or len(rows) >= total:
            break
        page += 1
    return rows, responses


def item_is_existing(
    item: dict[str, Any],
    existing: tuple[set[str], set[str], set[str]],
) -> bool:
    ssqd_codes, item_codes, names = existing
    return bool(
        str(item.get("ssqdCode") or "").strip() in ssqd_codes
        or str(item.get("ssqdBaseCode") or "").strip() in item_codes
        or normalize_name(item.get("ssqdName")) in names
    )


def detail_score(detail: dict[str, Any]) -> dict[str, int]:
    materials = [item for item in detail.get("clJson") or [] if isinstance(item, dict)]
    material_files = 0
    empty_forms = 0
    sample_files = 0
    for material in materials:
        enriched = material.get("json") if isinstance(material.get("json"), dict) else {}
        samples = enriched.get("materialSample") or json_value(material.get("fileSlybJson"), [])
        empties = enriched.get("materialEmpty") or json_value(material.get("fileKbbgJson"), [])
        sample_files += len(samples) if isinstance(samples, list) else 0
        empty_forms += len(empties) if isinstance(empties, list) else 0
    material_files = sample_files + empty_forms
    process_files = len(detail.get("processImgJson") or [])
    result_files = 0
    for result in detail.get("jgwjJson") or []:
        if not isinstance(result, dict):
            continue
        enriched = result.get("json") if isinstance(result.get("json"), dict) else {}
        files = enriched.get("resultSample") or json_value(result.get("fileJgwjJson"), [])
        result_files += len(files) if isinstance(files, list) else 0
    faq_count = len(detail.get("cjwtJson") or [])
    process_steps = len(detail.get("blhjJson") or [])
    body_chars = sum(
        len(plain_text(detail.get(key)))
        for key in ["acceptCondition", "hanleProc", "gbwsblsdName", "hanleAddr", "hanleTime"]
    )
    body_chars += sum(len(json.dumps(value, ensure_ascii=False)) for value in materials)
    score = (
        material_files * 12
        + empty_forms * 6
        + len(materials) * 5
        + result_files * 5
        + process_files * 3
        + faq_count * 2
        + process_steps * 2
        + min(body_chars // 1500, 20)
    )
    return {
        "score": score,
        "material_count": len(materials),
        "attachment_count": material_files + process_files + result_files,
        "sample_count": sample_files,
        "empty_form_count": empty_forms,
        "process_file_count": process_files,
        "result_file_count": result_files,
        "faq_count": faq_count,
        "process_step_count": process_steps,
    }


def fetch_candidate(item: dict[str, Any]) -> dict[str, Any]:
    client = session()
    body = request_json(
        client,
        DETAIL_API,
        {
            "ssqdCode": item.get("ssqdCode") or "",
            "ssqdId": item.get("id") or "",
        },
    )
    detail = body.get("data") if body.get("flag") in {True, "true"} else None
    if not isinstance(detail, dict) or not detail.get("ssqdName"):
        raise ValueError("detail response has no usable item")
    return {"list_item": item, "response": body, "detail": detail, "metrics": detail_score(detail)}


def rich_enough(candidate: dict[str, Any]) -> bool:
    metrics = candidate["metrics"]
    return bool(
        metrics["material_count"] >= 4
        and metrics["attachment_count"] >= 5
        and (candidate["detail"].get("acceptCondition") or candidate["detail"].get("hanleProc"))
    )


def candidate_theme(candidate: dict[str, Any]) -> str:
    return str(
        candidate["detail"].get("infoClassPtheme")
        or candidate["list_item"].get("infoClassPtheme")
        or "other"
    )


def candidate_agency(candidate: dict[str, Any]) -> str:
    return plain_text(
        candidate["detail"].get("implementOrgName")
        or candidate["detail"].get("ssqdDeptName")
        or "其他"
    )


def select_diverse(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """先保证主题覆盖，再在不显著牺牲丰富度的前提下补足数量。"""
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["metrics"]["score"],
            item["metrics"]["attachment_count"],
            item["metrics"]["material_count"],
            str(item["detail"].get("updateTime") or ""),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    theme_counts: dict[str, int] = {}
    agency_counts: dict[str, int] = {}

    # 第一轮：每个主题只保留一个代表事项，同一实施部门最多两个。
    for candidate in ranked:
        theme = candidate_theme(candidate)
        agency = candidate_agency(candidate)
        ssqd_code = str(candidate["detail"].get("ssqdCode") or "")
        if theme_counts.get(theme, 0) >= 1 or agency_counts.get(agency, 0) >= 2:
            continue
        selected.append(candidate)
        selected_ids.add(ssqd_code)
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
        agency_counts[agency] = agency_counts.get(agency, 0) + 1
        if len(selected) == count:
            return selected

    # 第二轮：主题最多两个、同一部门最多三个，用于主题数量不足时补足。
    for candidate in ranked:
        theme = candidate_theme(candidate)
        agency = candidate_agency(candidate)
        ssqd_code = str(candidate["detail"].get("ssqdCode") or "")
        if ssqd_code in selected_ids:
            continue
        if theme_counts.get(theme, 0) >= 2 or agency_counts.get(agency, 0) >= 3:
            continue
        selected.append(candidate)
        selected_ids.add(ssqd_code)
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
        agency_counts[agency] = agency_counts.get(agency, 0) + 1
        if len(selected) == count:
            return selected
    return selected


def file_items(material: dict[str, Any], key: str, fallback_key: str) -> list[dict[str, Any]]:
    enriched = material.get("json") if isinstance(material.get("json"), dict) else {}
    files = enriched.get(key)
    if not isinstance(files, list):
        files = json_value(material.get(fallback_key), [])
    return [item for item in files if isinstance(item, dict)]


def file_url(item: dict[str, Any]) -> str:
    value = str(item.get("filePath") or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    name = str(item.get("fileName") or "")
    return f"{BASE_URL}/intranet/{value.lstrip('/')}?attrname={quote(name)}"


def table(rows: list[tuple[str, Any]]) -> list[str]:
    output = ["| 字段 | 内容 |", "| --- | --- |"]
    output.extend(f"| {cell(name)} | {cell(value)} |" for name, value in rows)
    return output


def render_files(label: str, files: list[dict[str, Any]]) -> list[str]:
    if not files:
        return []
    output = [f"{label}："]
    for item in files:
        url = file_url(item)
        name = plain_text(item.get("fileName")) or "附件"
        output.append(f"- [{name}]({url})" if url else f"- {name}")
    return output


def render_guide(
    candidate: dict[str, Any],
    *,
    theme_name: str,
    policy_version: str,
) -> tuple[str, dict[str, Any]]:
    detail = candidate["detail"]
    metrics = candidate["metrics"]
    ssqd_code = str(detail.get("ssqdCode") or candidate["list_item"].get("ssqdCode") or "")
    ssqd_id = str(detail.get("id") or candidate["list_item"].get("id") or "")
    item_code = str(detail.get("ssqdBaseCode") or candidate["list_item"].get("ssqdBaseCode") or "")
    item_name = plain_text(detail.get("ssqdName"))
    document_id = f"ahzwfw_guide_{ssqd_code}"
    biz_type = f"ahzwfw_{plain_text(detail.get('baseCode')) or item_code}".lower()
    source_url = f"{DETAIL_PAGE}?ssqdId={ssqd_id}&ssqdCode={ssqd_code}&typeSipe=nameSkip"
    lines = [
        f"# {item_name}办事指南",
        "",
        f"- Document ID: {document_id}",
        "- Source Type: guide",
        f"- Biz Type: {biz_type}",
        "- Region Code: 340000000000",
        "- Region Name: 安徽省",
        f"- Policy Version: {policy_version}",
        "- Status: active",
        "- Effective Start Time: 2026-07-20T00:00:00+08:00",
        "- Effective End Time: 2027-07-19T23:59:59+08:00",
        "",
        "## 适用事项",
        "",
    ]
    lines.extend(
        table(
            [
                ("主题分类", theme_name),
                ("事项名称", item_name),
                ("目录清单名称", detail.get("mlqdName")),
                ("目录子项名称", detail.get("childMlqdName")),
                ("基本编码", detail.get("baseCode")),
                ("事项编码", item_code),
                ("事项类型", detail.get("ssqdTypeName")),
                ("服务对象", detail.get("serviceObjectName") or detail.get("serviceObjectTxt")),
                ("实施主体", detail.get("implementOrgName") or detail.get("ssqdDeptName")),
                ("实施主体性质", detail.get("ssqdSubjectNatureName")),
                ("行使层级", detail.get("exerciseLevelName")),
                ("权力来源", detail.get("ssqdOriginName")),
            ]
        )
    )
    lines.extend(["", "## 办理条件", "", plain_text(detail.get("acceptCondition")) or "以受理部门审核要求为准。"])
    lines.extend(["", "## 申请材料", ""])
    materials = [item for item in detail.get("clJson") or [] if isinstance(item, dict)]
    materials.sort(key=lambda item: int(item.get("materialSort") or 9999))
    if not materials:
        lines.append("官网当前未列出申请材料。")
    for index, material in enumerate(materials, 1):
        lines.extend([f"### {index}. {plain_text(material.get('materialName')) or '申请材料'}", ""])
        lines.extend(
            table(
                [
                    ("材料必要性", material.get("materialNeedName") or material.get("materialNeedTxt")),
                    ("材料类型", material.get("gbcllxTxt") or material.get("materialTypeName")),
                    ("材料形式", material.get("gbclxsTxt")),
                    ("来源渠道", material.get("sourceChannelName") or material.get("sourceChannelTxt")),
                    ("来源说明", material.get("sourceChannelDescription")),
                    ("材料要求", material.get("materialRequirement")),
                    ("受理标准", material.get("acceptStandard")),
                    ("填报须知", material.get("editNotice")),
                    ("材料依据", material.get("materialEvidence")),
                    ("纸质材料规格", material.get("paperMaterialsSpecifications")),
                ]
            )
        )
        samples = file_items(material, "materialSample", "fileSlybJson")
        empty_forms = file_items(material, "materialEmpty", "fileKbbgJson")
        if samples or empty_forms:
            lines.append("")
            lines.extend(render_files("样本文件", samples))
            lines.extend(render_files("空白表格", empty_forms))
        lines.append("")

    lines.extend(["## 办理方式", ""])
    lines.extend(
        table(
            [
                ("办理形式", detail.get("hanleFormName") or detail.get("hanleFormTxt")),
                ("网上办理能力", detail.get("gbwsblsdName")),
                ("材料提交方式", detail.get("materialCollectFormName")),
                ("办理地点", detail.get("hanleAddr")),
                ("办理时间", detail.get("hanleTime")),
                ("到现场次数", detail.get("runNum") or detail.get("toSceneTimes")),
                ("咨询电话", ", ".join(item.get("value", "") for item in json_value(detail.get("zxPhoneJson"), []))),
                ("监督投诉电话", ", ".join(item.get("value", "") for item in json_value(detail.get("tsPhoneJson"), []))),
            ]
        )
    )
    process_text = plain_text(detail.get("hanleProc"))
    if process_text:
        lines.extend(["", "### 办理流程说明", "", process_text])
    steps = [item for item in detail.get("blhjJson") or [] if isinstance(item, dict)]
    if steps:
        lines.extend(["", "### 办理环节", ""])
        for step in steps:
            lines.extend([f"#### {plain_text(step.get('blhjName')) or '办理环节'}", ""])
            lines.extend(
                table(
                    [
                        ("办理岗位", step.get("handlePostName")),
                        ("办理部门", step.get("handleDeptName")),
                        ("办理时限", f"{plain_text(step.get('handleTimeLimit'))}个工作日"),
                        ("岗位职责", step.get("handleDuty")),
                        ("审查内容", step.get("reviewContent")),
                    ]
                )
            )
            lines.append("")
    process_files = [item for item in detail.get("processImgJson") or [] if isinstance(item, dict)]
    if process_files:
        lines.extend(["### 流程图", ""])
        lines.extend(render_files("流程图文件", process_files))

    lines.extend(["", "## 办理时限", ""])
    lines.extend(
        table(
            [
                ("法定办结时限", f"{plain_text(detail.get('legalTime'))}个工作日"),
                ("承诺办结时限", f"{plain_text(detail.get('promiseTime'))}个工作日"),
                ("事项性质", detail.get("itemTypeName")),
                ("是否收费", detail.get("isChargeTxt")),
                ("收费标准", detail.get("chargeStandard")),
            ]
        )
    )

    lines.extend(["", "## 办理结果", ""])
    results = [item for item in detail.get("jgwjJson") or [] if isinstance(item, dict)]
    if results:
        for index, result in enumerate(results, 1):
            lines.extend([f"### {index}. {plain_text(result.get('resultName')) or '办理结果'}", ""])
            lines.extend(
                table(
                    [
                        ("结果类型", result.get("resultTypeTxt")),
                        ("有效期", result.get("validDate")),
                        ("领取方式", detail.get("resultReceiveWayName")),
                    ]
                )
            )
            files = file_items(result, "resultSample", "fileJgwjJson")
            if files:
                lines.append("")
                lines.extend(render_files("结果样本", files))
            lines.append("")
    else:
        lines.append(f"办理结果及领取方式：{plain_text(detail.get('resultReceiveWayName')) or '以办结通知为准。'}")

    lines.extend(["", "## 注意事项", ""])
    lines.extend(
        [
            "- 申请前应按官网材料清单逐项核对必要性、原件要求和电子文件清晰度。",
            "- 材料不齐全或不符合法定形式时，应按补齐补正通知一次性完成补正。",
            "- 办理地点、时间和电话可能调整，提交前应通过安徽政务服务网事项页面再次确认。",
        ]
    )
    faqs = [item for item in detail.get("cjwtJson") or [] if isinstance(item, dict)]
    if faqs:
        lines.extend(["", "## 常见问题", ""])
        for index, faq in enumerate(faqs, 1):
            lines.extend(
                [
                    f"### Q{index}: {plain_text(faq.get('problemTitle'))}",
                    "",
                    plain_text(faq.get("problemAnswer")) or "官网暂未提供答复。",
                    "",
                ]
            )

    markdown = "\n".join(lines).rstrip() + "\n"
    document = {
        "id": document_id,
        "source_type": "guide",
        "biz_type": biz_type,
        "region_code": "340000000000",
        "regionName": "安徽省",
        "policy_version": policy_version,
        "status": "active",
        "effectiveStartTime": "2026-07-20T00:00:00+08:00",
        "effectiveEndTime": "2027-07-19T23:59:59+08:00",
        "title": f"{item_name}办事指南",
        "item_name": item_name,
        "item_code": item_code,
        "ssqdCode": ssqd_code,
        "ssqdId": ssqd_id,
        "theme_code": str(detail.get("infoClassPtheme") or ""),
        "theme_name": theme_name,
        "official_source": True,
        "source_scope": "ahzwfw_official",
        "source_url": source_url,
        "detail_api": DETAIL_API,
        "evaluate_api": EVALUATE_API,
        "richness": metrics,
    }
    return markdown, document


def evaluate(client: requests.Session, ssqd_code: str) -> dict[str, Any]:
    return request_json(client, EVALUATE_API, {"serviceCode": ssqd_code})


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    output_dir = root / args.output
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = collect_existing(root)
    print(
        f"dedupe inventory: ssqd={len(existing[0])}, item_codes={len(existing[1])}, names={len(existing[2])}",
        flush=True,
    )

    client = session()
    themes, themes_response = fetch_themes(client)
    all_items, list_responses = fetch_list(client)
    candidates = [
        item
        for item in all_items
        if item.get("publishStatus") == "PUBLISH"
        and item.get("ssqdMasterStatusCode") == "USE"
        and not item_is_existing(item, existing)
    ]
    if args.candidate_limit > 0:
        candidates = candidates[: args.candidate_limit]
    print(f"detail candidates after dedupe: {len(candidates)}", flush=True)

    fetched: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(fetch_candidate, item): item for item in candidates}
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                fetched.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "ssqdCode": str(item.get("ssqdCode") or ""),
                        "ssqdName": plain_text(item.get("ssqdName")),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if index % 25 == 0 or index == len(futures):
                print(f"details: {index}/{len(futures)}, ok={len(fetched)}, failed={len(failures)}", flush=True)

    rich = [item for item in fetched if rich_enough(item)]
    selected = select_diverse(rich, args.count)
    if len(selected) != args.count:
        raise RuntimeError(f"only {len(selected)} rich non-duplicate guides found; expected {args.count}")

    policy_version = "ahzwfw-2026-07-20"
    manifest_documents: list[dict[str, Any]] = []
    selected_summary: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, 1):
        detail = candidate["detail"]
        theme_code = str(detail.get("infoClassPtheme") or candidate["list_item"].get("infoClassPtheme") or "")
        markdown, document = render_guide(
            candidate,
            theme_name=themes.get(theme_code, theme_code or "其他"),
            policy_version=policy_version,
        )
        ssqd_code = document["ssqdCode"]
        filename = f"guide_{ssqd_code}.md"
        raw_detail_name = f"guide_{ssqd_code}.json"
        raw_evaluate_name = f"guide_{ssqd_code}_evaluate.json"
        (output_dir / filename).write_text(markdown, encoding="utf-8")
        (raw_dir / raw_detail_name).write_text(
            json.dumps(candidate["response"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        evaluate_response = evaluate(client, ssqd_code)
        (raw_dir / raw_evaluate_name).write_text(
            json.dumps(evaluate_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        document["files"] = {
            "markdown": f"{args.output}/{filename}",
            "raw_detail": f"{args.output}/raw/{raw_detail_name}",
            "raw_evaluate": f"{args.output}/raw/{raw_evaluate_name}",
        }
        manifest_documents.append(document)
        selected_summary.append(
            {
                "rank": index,
                "ssqdCode": ssqd_code,
                "ssqdId": document["ssqdId"],
                "item_code": document["item_code"],
                "item_name": document["item_name"],
                "theme_code": theme_code,
                "theme_name": document["theme_name"],
                **candidate["metrics"],
            }
        )
        print(
            f"selected {index:02d}: {document['item_name']} "
            f"materials={candidate['metrics']['material_count']} "
            f"attachments={candidate['metrics']['attachment_count']} "
            f"score={candidate['metrics']['score']}",
            flush=True,
        )

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "从安徽政务服务网抓取的20个材料与附件较丰富且不重复的真实办事指南。",
        "source_page": SOURCE_PAGE,
        "source_theme_api": THEME_API,
        "source_list_api": LIST_API,
        "source_detail_api": DETAIL_API,
        "source_evaluate_api": EVALUATE_API,
        "selection_rule": "排除本地材料集和当前知识库已有事项后，优先保证主题多样性并限制同一实施部门占比，再按申请材料、样本、空白表格、流程图、结果样本和正文完整度综合排序。",
        "root": args.output,
        "documents": manifest_documents,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "themes.json").write_text(
        json.dumps(themes_response, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for index, body in enumerate(list_responses, 1):
        (raw_dir / f"list_page_{index}.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (raw_dir / "selected_items.json").write_text(
        json.dumps(selected_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_documents)} guides to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
