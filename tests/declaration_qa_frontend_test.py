from pathlib import Path


FRONTEND_ROOT = Path("frontend")


def frontend_text() -> str:
    files = [FRONTEND_ROOT / "index.html", *list((FRONTEND_ROOT / "src").rglob("*"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in files if path.is_file())


def test_declaration_qa_frontend_is_isolated_react_app():
    assert (FRONTEND_ROOT / "index.html").exists()
    assert (FRONTEND_ROOT / "src" / "App.jsx").exists()
    assert (FRONTEND_ROOT / "src" / "components" / "ChatPane.jsx").exists()
    assert (FRONTEND_ROOT / "src" / "styles.css").exists()

    combined = frontend_text()
    assert "/api/agents" not in combined
    assert "/api/declaration-qa" in combined


def test_frontend_uses_agentic_chat_without_ticket_binding_or_picker():
    combined = frontend_text()
    utils = (FRONTEND_ROOT / "src" / "utils.js").read_text(encoding="utf-8")

    assert "/chat/stream" in utils
    assert "/chat/confirm/stream" in utils
    assert "human_confirmation_required" in combined
    assert "请先处理上方人工确认" in combined
    assert "预约大厅" in combined
    assert "预约日期和时段" in combined
    assert "selected_slot_ids" in combined
    for label in [
        "查询申请事项",
        "查询申请详情",
        "查询预约记录",
        "查询预约大厅",
        "查询预约号源",
        "查询排队状态",
        "查询缴费订单",
        "查询电子票据",
        "查询申请回执",
        "查询办结结果",
        "查询送达状态",
        "查询咨询记录",
        "提交预约",
        "取消预约",
    ]:
        assert label in combined
    assert "/knowledge/documents" in utils
    assert "/sessions" in utils
    assert "/feedback" in utils
    assert "readSSE" in utils
    assert "FormData" in utils
    assert "chunking_strategy" in utils
    for legacy in [
        "/tickets",
        "applyId",
        "apply_id",
        "apply_context",
        "need_ticket_selection",
        "ticketPicker",
        "ticket-option",
        "current-ticket",
        "ticket-selection",
    ]:
        assert legacy not in combined


def test_frontend_waits_long_enough_for_rq_index_jobs():
    utils = (FRONTEND_ROOT / "src" / "utils.js").read_text(encoding="utf-8")
    app = (FRONTEND_ROOT / "src" / "App.jsx").read_text(encoding="utf-8")

    assert "export const INDEX_JOB_MAX_ATTEMPTS = 180" in utils
    assert "for (let i = 0; i < INDEX_JOB_MAX_ATTEMPTS; i++)" in app
    assert "chunksIndexed: job.payload?.chunks_indexed || 0" in app
