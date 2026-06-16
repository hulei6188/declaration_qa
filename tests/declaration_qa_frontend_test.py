from pathlib import Path


FRONTEND_ROOT = Path("frontend")


def test_declaration_qa_frontend_is_isolated_static_app():
    assert (FRONTEND_ROOT / "index.html").exists()
    assert (FRONTEND_ROOT / "src" / "app.js").exists()
    assert (FRONTEND_ROOT / "src" / "styles.css").exists()

    combined = "\n".join(path.read_text(encoding="utf-8") for path in FRONTEND_ROOT.rglob("*") if path.is_file())

    assert "frontend/src" not in combined
    assert "../frontend" not in combined
    assert "/api/agents" not in combined
    assert "/api/declaration-qa" in combined


def test_declaration_qa_frontend_has_required_workflows():
    html = (FRONTEND_ROOT / "index.html").read_text(encoding="utf-8")
    js = (FRONTEND_ROOT / "src" / "app.js").read_text(encoding="utf-8")

    for text in ["办件进度", "驳回原因", "材料清单", "当前工单", "更换", "清除", "ticketPicker", "知识入库", "切分方法", "知识版本管理", "RAG 质量评估", "审计与反馈"]:
        assert text in html
    for endpoint in [
        "/chat/stream",
        "/knowledge/documents",
        "/knowledge/upload",
        "/knowledge/uploads",
        "/rollout",
        "/versions",
        "/compare/",
        "/rollback",
        "/evaluation/datasets",
        "/evaluation/run-items/",
        "/index-jobs/",
        "/sessions",
        "/feedback",
        "/health",
        "/tickets",
    ]:
        assert endpoint in js
    assert "readSSE" in js
    assert "FormData" in js
    assert "chunking_strategy" in js
    assert 'type="button">上传并索引' in html
    assert "已选择，点击上传" in js
    assert "正在上传" in js
    assert "renderTicketSelection" in js
    assert "renderCurrentTicket" in js
    assert "changeCurrentTicket" in js
    assert "clearCurrentTicket" in js
    assert "openTicketPicker" in js
    assert "renderTicketPicker" in js
    assert "onQueryKeyDown" in js
    assert "requestSubmit" in js
    assert "need_ticket_selection" in js
    assert "ticket-option" in js


def test_frontend_waits_long_enough_for_rq_index_jobs():
    js = (FRONTEND_ROOT / "src" / "app.js").read_text(encoding="utf-8")

    assert "const INDEX_JOB_MAX_ATTEMPTS = 180" in js
    assert "chunksIndexed: payload.job_id ? null : payload.chunks_indexed ?? null" in js
