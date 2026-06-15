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

    for text in ["办件进度", "驳回原因", "材料清单", "知识入库", "知识版本管理", "RAG 质量评估", "审计与反馈"]:
        assert text in html
    for endpoint in [
        "/chat/stream",
        "/knowledge/documents",
        "/knowledge/upload",
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
    ]:
        assert endpoint in js
    assert "readSSE" in js
    assert "FormData" in js
