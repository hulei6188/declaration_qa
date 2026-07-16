import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar.jsx';
import ChatPane from './components/ChatPane.jsx';
import {
  DEFAULT_CONTEXT,
  buildChatRequest,
  startChatStream,
  resumeHumanConfirmationStream,
  readSSE,
  fetchSessions,
  fetchSession,
  fetchAgentRun,
  deleteSession,
  fetchDocuments,
  uploadTextDocument,
  uploadFileDocuments,
  setDocumentStatus,
  updateRollout,
  createVersion,
  compareVersion,
  rollbackDocument,
  fetchIndexJob,
  sendFeedback,
  etlStageLabel,
  compactJson,
  formatJobProgress,
  jobEtl,
  documentRevision,
  createToolTrace,
  reduceToolTrace,
  traceFromAgentRun,
  INDEX_JOB_MAX_ATTEMPTS,
  sleep,
} from './utils.js';

export default function App() {
  const [theme, setTheme] = useState(initialTheme);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(initialSidebarCollapsed);

  // ---- 请求上下文 ----
  const [ctx, setCtx] = useState(DEFAULT_CONTEXT);

  // ---- 聊天 ----
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [sendState, setSendState] = useState('等待输入');
  const assistantRef = useRef(null);

  // ---- 时间线与审计 ----
  const [timeline, setTimeline] = useState([]);
  const [audit, setAudit] = useState({ intent: '-', auditId: '-', refused: '-' });
  const [feedbackResult, setFeedbackResult] = useState('');

  // ---- 会话 ----
  const [sessions, setSessions] = useState([]);

  // ---- 文档 ----
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [latestVersionId, setLatestVersionId] = useState(null);
  const [docManageResult, setDocManageResult] = useState('选择文档后可发布、灰度、创建新版本或回滚。');

  // ---- 上传 ----
  const [uploadItems, setUploadItems] = useState([]);
  const [uploadResult, setUploadResult] = useState('支持单文件、批量文件和文本内容入库。');
  const [uploadInFlight, setUploadInFlight] = useState(false);

  // ---- 初始化 ----
  useEffect(() => {
    loadSessions();
    loadDocuments();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem('declaration-qa-theme', theme);
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem('declaration-qa-sidebar-collapsed', String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  // ---- 聊天 ----
  async function sendMessage(query, displayText, meta, ctxOverride, queryImage) {
    const userMsg = {
      role: 'user',
      content: displayText || query,
      attachments: queryImage ? [queryImage] : [],
    };
    setMessages((prev) => [...prev, userMsg]);
    const assistantMsg = { role: 'assistant', content: '', sources: [], toolTrace: createToolTrace() };
    setMessages((prev) => [...prev, assistantMsg]);
    assistantRef.current = '';
    setStreaming(true);
    setSendState('流式生成中…');
    setTimeline([]);
    setFeedbackResult('');

    try {
      const body = buildChatRequest(query, sessionId, ctxOverride || ctx, queryImage);
      const response = await startChatStream(body);
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      for await (const { event, data } of readSSE(response.body)) {
        if (['agent_decision', 'tool_executed', 'human_confirmation_required', 'human_confirmation_resolved', 'agent_observation', 'answer_generated', 'done', 'error'].includes(event)) {
          setMessages((prev) => updateLastAssistantTrace(prev, event, data));
        }
        if (event === 'human_confirmation_required') {
          setMessages((prev) => updateLastAssistantConfirmation(prev, data));
        } else if (event === 'run_paused') {
          setSessionId(data.session_id || sessionId);
        } else if (event === 'token') {
          assistantRef.current += (data.content || '');
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], content: assistantRef.current };
            return next;
          });
        } else if (event === 'sources') {
          const items = data.items || [];
          setMessages((prev) => attachSourcesToLastAssistant(prev, items));
          addTimeline(event, { count: (data.items || []).length });
        } else if (event === 'done') {
          setSessionId(data.session_id || sessionId);
          setAudit({
            intent: [data.route, data.intent].filter(Boolean).join(' / ') || '-',
            auditId: data.audit_id || '-',
            refused: data.refused ? (data.refuse_reason || '是') : '否',
          });
          addTimeline(event, data);
          loadSessions();
        } else {
          addTimeline(event, data);
        }
      }
    } catch (e) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          ...next[next.length - 1],
          content: (next[next.length - 1].content || '') + `\n请求失败：${e.message}`,
          toolTrace: reduceToolTrace(next[next.length - 1].toolTrace, 'error', { message: e.message }),
        };
        return next;
      });
      addTimeline('error', { message: e.message });
    } finally {
      setStreaming(false);
      setSendState('等待输入');
    }
  }

  async function resolveHumanConfirmation(messageIndex, confirmation, approved) {
    if (!confirmation?.interrupt_id || streaming) return;
    setStreaming(true);
    setSendState(approved ? '正在执行已确认操作…' : '正在取消操作…');
    setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => ({
      ...message,
      confirmation: { ...message.confirmation, status: 'submitting', decision: approved ? 'approved' : 'rejected' },
    })));
    let resumedText = '';
    try {
      const response = await resumeHumanConfirmationStream({
        session_id: confirmation.session_id || sessionId,
        interrupt_id: confirmation.interrupt_id,
        user_id: Number(ctx.userId || 1),
        approved: Boolean(approved),
      });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      for await (const { event, data } of readSSE(response.body)) {
        if (['tool_executed', 'human_confirmation_required', 'human_confirmation_resolved', 'agent_observation', 'answer_generated', 'done', 'error'].includes(event)) {
          setMessages((prev) => updateMessageTraceAt(prev, messageIndex, event, data));
        }
        if (event === 'human_confirmation_required') {
          setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => ({
            ...message,
            confirmation: { ...data, status: 'pending' },
          })));
        } else if (event === 'run_paused') {
          setSessionId(data.session_id || sessionId);
        } else if (event === 'human_confirmation_resolved') {
          setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => ({
            ...message,
            confirmation: {
              ...message.confirmation,
              status: data.approved ? 'approved' : 'rejected',
              decision: data.approved ? 'approved' : 'rejected',
            },
          })));
        } else if (event === 'token') {
          resumedText += data.content || '';
          setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => ({ ...message, content: resumedText })));
        } else if (event === 'sources') {
          setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => ({ ...message, sources: data.items || [] })));
        } else if (event === 'done') {
          setSessionId(data.session_id || sessionId);
          setAudit({
            intent: [data.route, data.intent].filter(Boolean).join(' / ') || '-',
            auditId: data.audit_id || '-',
            refused: data.refused ? (data.refuse_reason || '是') : '否',
          });
          loadSessions();
        } else if (event === 'error') {
          throw new Error(data.message || data.error || '人工确认处理失败');
        }
        addTimeline(event, data);
      }
    } catch (e) {
      setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => ({
        ...message,
        confirmation: { ...message.confirmation, status: 'failed', error: e.message },
      })));
      addTimeline('error', { message: e.message });
    } finally {
      setStreaming(false);
      setSendState('等待输入');
    }
  }

  function addTimeline(event, data) {
    setTimeline((prev) => [{ event, data: compactJson(data) }, ...prev]);
  }

  // ---- 会话 ----
  async function loadSessions() {
    try {
      const list = await fetchSessions(ctx.userId);
      setSessions(list || []);
    } catch { /* 忽略 */ }
  }

  async function loadSession(sid) {
    try {
      const session = await fetchSession(sid, ctx.userId);
      setSessionId(session.session_id);
      setMessages(
        (session.messages || []).map((m) => ({
          role: m.role === 'user' ? 'user' : 'assistant',
          content: m.content || '',
          attachments: m.role === 'user' ? (m.attachments || []) : [],
          sources: m.role === 'user' ? [] : (m.sources || []),
          runId: m.role === 'user' ? '' : (m.run_id || ''),
          toolTrace: null,
        }))
      );
      setTimeline([]);
      setAudit({ intent: '-', auditId: '-', refused: '-' });
      setFeedbackResult('');

      addTimeline('session_loaded', { session_id: session.session_id, message_count: session.message_count });
    } catch (e) {
      addTimeline('session_error', { message: `HTTP ${e.message}` });
    }
  }

  async function loadToolTrace(messageIndex, runId) {
    if (!runId) return;
    setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => (
      message.runId !== runId
        ? message
        : { ...message, toolTrace: { ...createToolTrace('loading'), loaded: false } }
    )));
    try {
      const run = await fetchAgentRun(runId, ctx.userId);
      setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => (
        message.runId !== runId ? message : { ...message, toolTrace: traceFromAgentRun(run) }
      )));
    } catch (error) {
      setMessages((prev) => updateMessageAt(prev, messageIndex, (message) => (
        message.runId !== runId
          ? message
          : {
            ...message,
            toolTrace: {
              ...createToolTrace('failed'),
              loaded: false,
              error: `执行记录读取失败：${error.message}`,
            },
          }
      )));
    }
  }

  async function removeSession(sid) {
    const target = sessions.find((item) => item.session_id === sid);
    const label = target?.title || '该会话';
    if (!window.confirm(`确定删除「${label}」？`)) return;
    try {
      await deleteSession(sid, ctx.userId);
      setSessions((prev) => prev.filter((item) => item.session_id !== sid));
      if (sid === sessionId) {
        clearChat();
      }
      addTimeline('session_deleted', { session_id: sid });
    } catch (e) {
      addTimeline('session_delete_error', { session_id: sid, message: e.message });
    }
  }

  // ---- 文档 ----
  async function loadDocuments() {
    try {
      const docs = await fetchDocuments();
      setDocuments(docs || []);
    } catch { /* 忽略 */ }
  }

  function selectDocument(docId) {
    setSelectedDocId(docId);
    const doc = documents.find((d) => d.document_id === docId);
    if (doc) {
      const revision = documentRevision(doc);
      setDocManageResult(`已选择：${doc.title} / ${etlStageLabel(revision.status)} / v${doc.version_no} / ${revision.revision}`);
    }
  }

  async function doSetDocumentStatus(action) {
    if (!selectedDocId) { setDocManageResult('请先选择文档。'); return; }
    try {
      const res = await setDocumentStatus(selectedDocId, action);
      const payload = await res.json();
      setDocManageResult(`状态已更新：${payload.status}`);
      await loadDocuments();
    } catch (e) {
      setDocManageResult(`状态更新失败：${e.message}`);
    }
  }

  async function doUpdateRollout() {
    if (!selectedDocId) { setDocManageResult('请先选择文档。'); return; }
    const percent = document.getElementById('rolloutPercent')?.value || 100;
    try {
      const res = await updateRollout(selectedDocId, percent);
      const payload = await res.json();
      setDocManageResult(`灰度已更新：${payload.rollout_percent}%`);
      await loadDocuments();
    } catch (e) {
      setDocManageResult(`灰度更新失败：${e.message}`);
    }
  }

  async function doCreateVersion(content) {
    if (!selectedDocId) { setDocManageResult('请先选择文档。'); return; }
    if (!content) { setDocManageResult('请填写新版本内容。'); return; }
    try {
      const res = await createVersion(selectedDocId, content);
      const payload = await res.json();
      setLatestVersionId(payload.document_id);
      setDocManageResult(`已创建 draft 版本：v${payload.version_no}`);
      if (payload.job_id) await fetchIndexJob(payload.job_id);
      await loadDocuments();
    } catch (e) {
      setDocManageResult(`创建版本失败：${e.message}`);
    }
  }

  async function doCompareVersion() {
    if (!selectedDocId || !latestVersionId) {
      setDocManageResult('请先选择文档并创建新版本。');
      return;
    }
    try {
      const payload = await compareVersion(selectedDocId, latestVersionId);
      setDocManageResult(`对比完成：新增 ${payload.added.length} 条，移除 ${payload.removed.length} 条。`);
    } catch (e) {
      setDocManageResult(`对比失败：${e.message}`);
    }
  }

  async function doRollback() {
    if (!selectedDocId) { setDocManageResult('请先选择要回滚到的版本。'); return; }
    try {
      const res = await rollbackDocument(selectedDocId);
      const payload = await res.json();
      setDocManageResult(`已回滚并发布：${payload.document_id}`);
      await loadDocuments();
    } catch (e) {
      setDocManageResult(`回滚失败：${e.message}`);
    }
  }

  // ---- 上传 ----
  async function doUpload(title, content, files, chunkingStrategy, policyVersion) {
    if (uploadInFlight) return;
    if (!files.length && !content) {
      setUploadResult('请选择文件，或粘贴文档内容。');
      return;
    }
    if (files.length) {
      await uploadBatch(files, title, chunkingStrategy, policyVersion);
      return;
    }
    setUploadResult('文本入库中');
    try {
      const res = await uploadTextDocument(
        title || content.split('\n')[0].trim().slice(0, 80),
        content,
        ctx.regionCode,
        policyVersion || ctx.policyVersion,
        chunkingStrategy
      );
      const payload = await res.json();
      if (payload.job_id) {
        setUploadResult(`索引任务已提交：${payload.document_id}`);
        const job = await waitForJob(payload.job_id);
        if (job.status === 'succeeded') {
          setUploadResult(`已索引 ${job.payload?.chunks_indexed || payload.chunks_indexed || 0} 个分块 · ${formatJobProgress(job)}`);
        } else if (job.status === 'failed') {
          throw new Error(job.error_message || '索引任务失败');
        }
      } else {
        setUploadResult(`已索引 ${payload.chunks_indexed} 个分块：${payload.document_id}`);
      }
      await loadDocuments();
    } catch (e) {
      setUploadResult(`上传失败：${e.message}`);
    }
  }

  async function uploadBatch(files, title, strategy, policyVersion) {
    const label = document.getElementById('chunkingStrategy')?.selectedOptions[0]?.textContent || 'Chonkie 结构分块';
    const items = files.map((f, i) => ({
      id: `${f.name}-${f.size}-${f.lastModified}-${i}`,
      name: f.name,
      size: f.size,
      status: 'waiting',
      message: '准备上传',
      strategyLabel: label,
      chunksIndexed: null,
      progress: 0,
      etl: null,
      jobId: '',
      documentId: '',
    }));
    setUploadItems(items);
    setUploadInFlight(true);
    setUploadResult(`准备上传 ${items.length} 个文件`);

    try {
      setUploadItems((prev) => prev.map((it) => ({ ...it, status: 'running', message: '上传中' })));
      const res = await uploadFileDocuments(files, title, ctx.regionCode, policyVersion || ctx.policyVersion, strategy);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const batch = await res.json();
      const docs = batch.documents || [];

      setUploadItems((prev) =>
        prev.map((it, i) => {
          const payload = docs[i];
          if (!payload) return it;
          return {
            ...it,
            status: 'indexing',
            message: payload.job_id ? '索引任务已提交' : '索引完成',
            jobId: payload.job_id || '',
            documentId: payload.document_id || '',
          };
        })
      );

      let failedCount = 0;
      for (let i = 0; i < docs.length; i++) {
        const payload = docs[i];
        if (!payload?.job_id) continue;
        const job = await waitForJob(payload.job_id, i);
        if (['failed', 'cancelled'].includes(job.status)) failedCount += 1;
        setUploadItems((prev) =>
          prev.map((it, idx) => {
            if (idx !== i) return it;
            if (['failed', 'cancelled'].includes(job.status)) {
              return {
                ...it,
                status: 'failed',
                message: job.error_message || '索引任务失败',
                progress: job.progress || it.progress || 0,
                etl: jobEtl(job),
              };
            }
            if (job.status === 'succeeded') {
              return {
                ...it,
                status: 'succeeded',
                message: '已入库',
                chunksIndexed: job.payload?.chunks_indexed || 0,
                progress: 100,
                etl: jobEtl(job),
              };
            }
            return {
              ...it,
              status: 'pending',
              message: formatJobProgress(job),
              progress: job.progress || 0,
              etl: jobEtl(job),
            };
          })
        );
      }

      if (failedCount) throw new Error(`失败 ${failedCount}/${items.length}`);
      setUploadResult(`批量入库完成`);
      await loadDocuments();
    } catch (e) {
      setUploadItems((prev) =>
        prev.map((it) => (['waiting', 'running', 'indexing'].includes(it.status) ? { ...it, status: 'failed', message: e.message } : it))
      );
      setUploadResult(`批量入库失败：${e.message}`);
    } finally {
      setUploadInFlight(false);
    }
  }

  async function waitForJob(jobId, uploadIdx) {
    for (let i = 0; i < INDEX_JOB_MAX_ATTEMPTS; i++) {
      try {
        const job = await fetchIndexJob(jobId);
        const summary = formatJobProgress(job);
        if (uploadIdx !== undefined && uploadIdx !== null) {
          setUploadItems((prev) =>
            prev.map((item, idx) => (
              idx === uploadIdx
                ? { ...item, message: summary, progress: job.progress || 0, etl: jobEtl(job) }
                : item
            ))
          );
        }
        if (['succeeded', 'failed', 'cancelled'].includes(job.status)) return job;
        setUploadResult(summary);
      } catch { /* 重试 */ }
      await sleep(1000);
    }
    return { status: 'pending', progress: 0, payload: {} };
  }

  // ---- 反馈 ----
  async function doFeedback(rating) {
    if (!audit.auditId || audit.auditId === '-') {
      setFeedbackResult('暂无审计 ID。');
      return;
    }
    try {
      await sendFeedback(audit.auditId, rating);
      setFeedbackResult(rating === 'up' ? '已记录：有帮助' : '已记录：需改进');
    } catch (e) {
      setFeedbackResult(`反馈失败：${e.message}`);
    }
  }

  // ---- 渲染 ----
  return (
    <div className="app-shell" data-theme={theme}>
      <main className={`workspace${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}>
        <Sidebar
          sessions={sessions}
          documents={documents}
          selectedDocId={selectedDocId}
          uploadItems={uploadItems}
          uploadResult={uploadResult}
          uploadInFlight={uploadInFlight}
          docManageResult={docManageResult}
          onRefreshSessions={loadSessions}
          onLoadSession={loadSession}
          onDeleteSession={removeSession}
          onUpload={doUpload}
          onSelectDocument={selectDocument}
          onRefreshDocuments={loadDocuments}
          onSetDocumentStatus={doSetDocumentStatus}
          onUpdateRollout={doUpdateRollout}
          onCreateVersion={doCreateVersion}
          onCompareVersion={doCompareVersion}
          onRollback={doRollback}
          onCollapse={() => setSidebarCollapsed(true)}
        />
        <ChatPane
          messages={messages}
          streaming={streaming}
          sendState={sendState}
          sessionId={sessionId}
          onSend={sendMessage}
          ctx={ctx}
          setCtx={setCtx}
          audit={audit}
          feedbackResult={feedbackResult}
          onFeedback={doFeedback}
          onLoadToolTrace={loadToolTrace}
          onResolveConfirmation={resolveHumanConfirmation}
          theme={theme}
          onToggleTheme={() => setTheme((value) => (value === 'dark' ? 'light' : 'dark'))}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed((value) => !value)}
        />
      </main>
    </div>
  );
}

function initialTheme() {
  const saved = window.localStorage.getItem('declaration-qa-theme');
  if (saved === 'light' || saved === 'dark') return saved;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function initialSidebarCollapsed() {
  const saved = window.localStorage.getItem('declaration-qa-sidebar-collapsed');
  if (saved === 'true' || saved === 'false') return saved === 'true';
  return window.matchMedia?.('(max-width: 860px)').matches ?? false;
}

function attachSourcesToLastAssistant(messages, sources) {
  const next = [...messages];
  for (let index = next.length - 1; index >= 0; index--) {
    if (next[index]?.role !== 'assistant') continue;
    next[index] = { ...next[index], sources: sources || [] };
    break;
  }
  return next;
}

function updateLastAssistantTrace(messages, event, data) {
  const next = [...messages];
  for (let index = next.length - 1; index >= 0; index -= 1) {
    if (next[index]?.role !== 'assistant') continue;
    const toolTrace = reduceToolTrace(next[index].toolTrace, event, data);
    next[index] = {
      ...next[index],
      runId: toolTrace.runId || next[index].runId || '',
      toolTrace,
    };
    break;
  }
  return next;
}

function updateLastAssistantConfirmation(messages, confirmation) {
  const next = [...messages];
  for (let index = next.length - 1; index >= 0; index -= 1) {
    if (next[index]?.role !== 'assistant') continue;
    next[index] = { ...next[index], confirmation: { ...confirmation, status: 'pending' } };
    break;
  }
  return next;
}

function updateMessageTraceAt(messages, index, event, data) {
  return updateMessageAt(messages, index, (message) => ({
    ...message,
    toolTrace: reduceToolTrace(message.toolTrace, event, data),
  }));
}

function updateMessageAt(messages, index, updater) {
  if (index < 0 || index >= messages.length) return messages;
  const next = [...messages];
  next[index] = updater(next[index]);
  return next;
}
