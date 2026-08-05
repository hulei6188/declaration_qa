from pathlib import Path


FRONTEND_ROOT = Path("frontend")


def frontend_text() -> str:
    files = [FRONTEND_ROOT / "index.html", *list((FRONTEND_ROOT / "src").rglob("*"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in files if path.is_file())


def test_gov_service_agent_frontend_is_isolated_react_app():
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
    assert "查看资料检索" in combined
    assert "查看业务查询" in combined
    assert "正在查询办事资料" in combined
    assert "查看办理进度" not in combined
    assert "查看工具调用过程" not in combined
    assert "正在调用工具" not in combined
    assert "Agent 工具执行过程" not in combined
    assert "预约大厅" in combined
    assert "预约日期和时段" in combined
    assert "selected_slot_ids" in combined
    for label in [
        "查询办事资料",
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


def test_frontend_displays_empty_tool_results_without_failure_state():
    utils = (FRONTEND_ROOT / "src" / "utils.js").read_text(encoding="utf-8")
    timeline = (FRONTEND_ROOT / "src" / "components" / "ToolCallTimeline.jsx").read_text(encoding="utf-8")
    styles = (FRONTEND_ROOT / "src" / "styles.css").read_text(encoding="utf-8")

    assert "if (status === 'not_found') return 'empty'" in utils
    assert "status: toolTraceStatus(data.ok, data.status)" in utils
    assert "status: toolTraceStatus(call.ok, call.status)" in utils
    assert "if (status === 'empty') return '无记录'" in timeline
    assert ".status-empty .tool-step-icon" in styles


def test_sidebar_collapse_uses_progressive_motion_with_reduced_motion_fallback():
    styles = (FRONTEND_ROOT / "src" / "styles.css").read_text(encoding="utf-8")

    assert "transition: grid-template-columns .28s" in styles
    assert ".workspace.sidebar-collapsed .context-pane.app-sidebar" in styles
    assert "transform: translateX(-100%)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles


def test_composer_only_shows_real_streaming_phases():
    app = (FRONTEND_ROOT / "src" / "App.jsx").read_text(encoding="utf-8")
    chat = (FRONTEND_ROOT / "src" / "components" / "ChatPane.jsx").read_text(encoding="utf-8")

    assert "等待输入" not in app + chat
    assert "streamPhaseLabel" in app
    assert "正在连接服务" in app
    assert "正在生成回答" in app
    assert "sendState &&" in chat


def test_failed_form_request_retains_current_dialog_task_and_uploaded_values():
    app = (FRONTEND_ROOT / "src" / "App.jsx").read_text(encoding="utf-8")
    utils = (FRONTEND_ROOT / "src" / "utils.js").read_text(encoding="utf-8")

    assert "latestCollectingDialogTask(messages)" in app
    assert "retainSubmittedFormValue(activeDialogTask, meta?.formSubmission)" in app
    assert "dialogTask: retainedDialogTask" in app
    assert "[fieldName]: submission.value" in utils
    assert "confirmation: m.role === 'user' ? null : (m.confirmation || null)" in app


def test_completed_dialog_task_blocks_stale_collecting_form_resurrection():
    utils = (FRONTEND_ROOT / "src" / "utils.js").read_text(encoding="utf-8")

    assert "if (task.status === 'collecting' && task.next_field?.name) return task;" in utils
    assert "return {};" in utils
    assert "Object.keys(task).length === 0" in utils


def test_confirmation_ready_event_hides_completed_form():
    app = (FRONTEND_ROOT / "src" / "App.jsx").read_text(encoding="utf-8")
    utils = (FRONTEND_ROOT / "src" / "utils.js").read_text(encoding="utf-8")

    assert "confirmationCompletesFormCollection(confirmation) ? {}" in app
    assert "['application_submit', 'complaint_create'].includes" in utils


def test_frontend_hides_resolved_confirmation_cards_after_resume():
    app = (FRONTEND_ROOT / "src" / "App.jsx").read_text(encoding="utf-8")
    chat = (FRONTEND_ROOT / "src" / "components" / "ChatPane.jsx").read_text(encoding="utf-8")

    assert "confirmation: null" in app
    assert "confirmationCardIsVisible(m.confirmation)" in chat
    assert "['pending', 'submitting', 'failed'].includes(status)" in chat
