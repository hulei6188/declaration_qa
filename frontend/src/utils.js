// ---- 常量 ----
export const API_BASE = '';
export const INDEX_JOB_MAX_ATTEMPTS = 180;

export const DEFAULT_CONTEXT = {
  userId: 1,
  regionCode: '340000',
  bizType: '',
  appVersion: '2.0.0',
  policyVersion: '2025',
};

export function latestCollectingDialogTask(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const task = messages[index]?.dialogTask;
    if (!task || typeof task !== 'object' || Object.keys(task).length === 0) continue;
    if (task.status === 'collecting' && task.schema?.steps?.length) return task;
    // 最近的非空任务已经进入 ready/completed/cancelled 等后续状态时，
    // 不得继续向前寻找并复活同一会话中的旧 collecting 表单。
    return {};
  }
  return {};
}

export function retainSubmittedFormValues(task, submission) {
  const values = submission?.values;
  if (!values || typeof values !== 'object' || Array.isArray(values) || task?.status !== 'collecting') return task || {};
  return {
    ...task,
    values: {
      ...(task.values || {}),
      ...values,
    },
  };
}

export function confirmationCompletesFormCollection(confirmation) {
  if (confirmation?.form_ready_for_confirmation) return true;
  return (confirmation?.actions || []).some((action) =>
    ['application_submit', 'complaint_create'].includes(String(action?.tool_name || ''))
  );
}

// ---- 格式化工具 ----
export function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

export function formatBytes(value) {
  const size = Number(value || 0);
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

export function formatMetric(value) {
  if (value === null || value === undefined) return '-';
  return `${Math.round(Number(value) * 100)}%`;
}

export function compactJson(value) {
  return JSON.stringify(value || {}, null, 2).slice(0, 900);
}

export const ETL_STAGE_LABELS = {
  waiting: '等待',
  pending: '排队中',
  accepted: '已接收',
  extract: '解析',
  metadata: '元数据',
  transform: '层级切块',
  embedding_load: '向量入库',
  atomic_publish: '原子发布',
  indexing: '入库中',
  published: '已发布',
  draft: '草稿',
  succeeded: '已完成',
  reindex: '重建索引',
  delete: '删除索引',
  cancelled: '已取消',
  failed: '失败',
};

export function etlStageLabel(stage) {
  return ETL_STAGE_LABELS[String(stage || '')] || String(stage || '等待');
}

export function jobEtl(job) {
  return job?.payload?.etl || {};
}

export function formatJobProgress(job) {
  const etl = jobEtl(job);
  const stage = etl.current_stage || job?.status || 'pending';
  const label = etlStageLabel(stage);
  const progress = Number(job?.progress || 0);
  const message = etl.last_message || job?.message || '';
  return `${label} ${progress}%${message ? ` · ${message}` : ''}`;
}

export function documentRevision(doc) {
  const metadata = doc?.metadata || {};
  return {
    revision: metadata.knowledge_revision || '-',
    stage: metadata.revision_stage || metadata.doc_status || doc?.status || '-',
    visible: metadata.revision_visible,
    status: doc?.status || metadata.doc_status || '-',
  };
}

export async function copyText(value) {
  const text = String(value ?? '');
  if (!text) return false;

  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // 非安全上下文中继续使用文本框兜底方案。
    }
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.inset = '0 auto auto 0';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);
  try {
    return document.execCommand('copy');
  } finally {
    document.body.removeChild(textarea);
  }
}

// ---- 接口 ----
async function request(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res;
}

async function requestJson(url, options) {
  const res = await request(url, options);
  return res.json();
}

// 聊天服务端事件流
export function buildChatRequest(query, sessionId, ctx, queryImage, formSubmission) {
  const payload = {
    query,
    user_id: Number(ctx.userId || 1),
    region_code: ctx.regionCode || '340000',
    app_version: ctx.appVersion,
    policy_version: ctx.policyVersion || '2025',
    session_id: sessionId,
  };
  if (ctx.bizType) payload.biz_type = ctx.bizType;
  if (formSubmission?.values && typeof formSubmission.values === 'object') payload.form_submission = formSubmission;
  if (queryImage?.data_url) {
    payload.query_image = {
      filename: queryImage.filename || '',
      content_type: queryImage.content_type || 'image/png',
      size_bytes: Number(queryImage.size_bytes || 0),
      data_url: queryImage.data_url,
      text_present: Boolean(queryImage.text_present),
    };
  }
  return payload;
}

export async function startChatStream(body) {
  return fetch(`${API_BASE}/api/declaration-qa/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function resumeHumanConfirmationStream(body) {
  return fetch(`${API_BASE}/api/declaration-qa/chat/confirm/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// 会话
export function fetchSessions(userId) {
  return requestJson(`/api/declaration-qa/sessions?user_id=${encodeURIComponent(userId || 1)}`);
}

export function fetchSession(sessionId, userId) {
  return requestJson(`/api/declaration-qa/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId || 1)}`);
}

export function fetchAgentRun(runId, userId) {
  return requestJson(`/api/declaration-qa/agent-runs/${encodeURIComponent(runId)}?user_id=${encodeURIComponent(userId || 1)}`);
}

export function deleteSession(sessionId, userId) {
  return requestJson(`/api/declaration-qa/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId || 1)}`, {
    method: 'DELETE',
  });
}

// 文档
export function fetchDocuments() {
  return requestJson('/api/declaration-qa/knowledge/documents');
}

export function uploadTextDocument(title, content, regionCode, policyVersion, chunkingStrategy) {
  return request('/api/declaration-qa/knowledge/documents', {
    method: 'POST',
    body: JSON.stringify({
      title,
      content,
      region_code: regionCode || '340000',
      policy_version: policyVersion || '2025',
      chunking_strategy: chunkingStrategy,
    }),
  });
}

export function uploadFileDocuments(files, title, regionCode, policyVersion, chunkingStrategy) {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  form.append('title', title || '');
  form.append('region_code', regionCode || '340000');
  form.append('policy_version', policyVersion || '2025');
  form.append('chunking_strategy', chunkingStrategy);
  return fetch(`${API_BASE}/api/declaration-qa/knowledge/uploads`, { method: 'POST', body: form });
}

export function setDocumentStatus(documentId, action) {
  return request(`/api/declaration-qa/knowledge/documents/${documentId}/${action}`, { method: 'POST' });
}

export function updateRollout(documentId, percent) {
  return request(`/api/declaration-qa/knowledge/documents/${documentId}/rollout`, {
    method: 'POST',
    body: JSON.stringify({ rollout_percent: Number(percent || 0) }),
  });
}

export function createVersion(documentId, content) {
  return request(`/api/declaration-qa/knowledge/documents/${documentId}/versions`, {
    method: 'POST',
    body: JSON.stringify({ content, change_note: '前端创建新版本' }),
  });
}

export function compareVersion(documentId, versionId) {
  return requestJson(`/api/declaration-qa/knowledge/documents/${documentId}/compare/${versionId}`);
}

export function rollbackDocument(targetDocumentId) {
  return request(`/api/declaration-qa/knowledge/documents/rollback`, {
    method: 'POST',
    body: JSON.stringify({ target_document_id: targetDocumentId, change_note: '前端回滚' }),
  });
}

// 索引任务
export function fetchIndexJob(jobId) {
  return requestJson(`/api/declaration-qa/index-jobs/${jobId}`);
}

// 反馈
export function sendFeedback(auditId, rating) {
  return request('/api/declaration-qa/feedback', {
    method: 'POST',
    body: JSON.stringify({ audit_id: auditId, rating, comment: '' }),
  });
}

// 评估
export function createDataset(name, cases) {
  return request('/api/declaration-qa/evaluation/datasets', {
    method: 'POST',
    body: JSON.stringify({ name, description: '前端创建', cases }),
  });
}

export function runEvaluation(datasetId) {
  return request(`/api/declaration-qa/evaluation/datasets/${datasetId}/runs`, {
    method: 'POST',
    body: JSON.stringify({ note: 'frontend run', top_k: 4 }),
  });
}

export function annotateEvalItem(itemId, label) {
  return request(`/api/declaration-qa/evaluation/run-items/${itemId}/annotation`, {
    method: 'POST',
    body: JSON.stringify({ label, comment: '前端人工标注' }),
  });
}

// 服务端事件流辅助函数
export function parseSSE(raw) {
  const event = { event: 'message', data: {} };
  raw.split('\n').forEach((line) => {
    if (line.startsWith('event:')) event.event = line.slice(6).trim();
    if (line.startsWith('data:')) {
      const text = line.slice(5).trim();
      try { event.data = JSON.parse(text); }
      catch { event.data = { text }; }
    }
  });
  return event;
}

export async function* readSSE(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';
      for (const part of parts) yield parseSSE(part);
    }
    if (buffer.trim()) yield parseSSE(buffer);
  } finally {
    // The workflow emits an explicit terminal event. If the consumer stops at
    // that event, cancel the body instead of waiting for a delayed TCP EOF.
    try { await reader.cancel(); }
    catch { /* The server may already have closed the stream. */ }
    reader.releaseLock();
  }
}

export async function uploadApplicationMaterial(file, payload) {
  const body = new FormData();
  body.append('file', file);
  body.append('session_id', payload.sessionId);
  body.append('user_id', String(payload.userId || 1));
  body.append('item_code', payload.itemCode || '');
  body.append('field_name', payload.fieldName);
  body.append('material_key', payload.materialKey);
  const response = await fetch(`${API_BASE}/api/declaration-qa/application-materials/upload`, {
    method: 'POST',
    body,
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (data?.detail) message = data.detail;
    } catch { /* 保留 HTTP 状态 */ }
    throw new Error(message);
  }
  return response.json();
}

export function createToolTrace(status = 'running') {
  return {
    status,
    phase: status === 'running' ? 'planning' : status,
    loaded: status === 'running',
    tools: [],
    observation: null,
    error: '',
  };
}

export function fetchEvaluationDatasets() {
  return requestJson('/api/declaration-qa/evaluation/datasets');
}

export function fetchEvaluationRuns(datasetId) {
  return requestJson(`/api/declaration-qa/evaluation/datasets/${encodeURIComponent(datasetId)}/runs`);
}

export function fetchEvaluationRun(runId) {
  return requestJson(`/api/declaration-qa/evaluation/runs/${encodeURIComponent(runId)}`);
}

export function publishEvaluationDataset(datasetId) {
  return requestJson(`/api/declaration-qa/evaluation/datasets/${encodeURIComponent(datasetId)}/publish`, { method: 'POST' });
}

export function fetchEvaluationGate(runId) {
  return requestJson(`/api/declaration-qa/evaluation/runs/${encodeURIComponent(runId)}/gate`);
}

export function compareEvaluationRuns(runId, baselineRunId) {
  return requestJson(`/api/declaration-qa/evaluation/runs/${encodeURIComponent(runId)}/compare/${encodeURIComponent(baselineRunId)}`);
}

export function fetchEvaluationCandidates(status = '') {
  const suffix = status ? `?status=${encodeURIComponent(status)}` : '';
  return requestJson(`/api/declaration-qa/evaluation/candidates${suffix}`);
}

export function mineEvaluationCandidates() {
  return requestJson('/api/declaration-qa/evaluation/candidates/mine', { method: 'POST' });
}

export function reviewEvaluationCandidate(candidateId, payload) {
  return requestJson(`/api/declaration-qa/evaluation/candidates/${encodeURIComponent(candidateId)}/review`, {
    method: 'POST', body: JSON.stringify(payload),
  });
}

export function promoteEvaluationCandidate(candidateId, datasetId, expected) {
  return requestJson(`/api/declaration-qa/evaluation/candidates/${encodeURIComponent(candidateId)}/promote`, {
    method: 'POST', body: JSON.stringify({ dataset_id: datasetId, expected }),
  });
}

export function createHumanServiceTicket(auditId, sessionId, userId, note = '') {
  return requestJson(humanServiceTicketsPath(), {
    method: 'POST',
    body: JSON.stringify({ audit_id: auditId, session_id: sessionId || '', user_id: Number(userId || 1), note }),
  });
}

export function fetchHumanServiceTickets(status = '') {
  const suffix = status ? `?status=${encodeURIComponent(status)}` : '';
  return requestJson(`${humanServiceTicketsPath()}${suffix}`);
}

export function claimHumanServiceTicket(ticketId) {
  return requestJson(humanServiceTicketActionPath(ticketId, 'claim'), { method: 'POST' });
}

export function updateHumanServiceTicketStatus(ticketId, status) {
  return requestJson(humanServiceTicketActionPath(ticketId, 'status'), {
    method: 'POST', body: JSON.stringify({ status }),
  });
}

export function sendHumanServiceMessage(ticketId, content) {
  return requestJson(humanServiceTicketActionPath(ticketId, 'messages'), {
    method: 'POST', body: JSON.stringify({ content }),
  });
}

function humanServiceTicketsPath() {
  return ['/api/declaration-qa/human-service', 'tickets'].join('/');
}

function humanServiceTicketActionPath(ticketId, action) {
  return [humanServiceTicketsPath(), encodeURIComponent(ticketId), action].join('/');
}

export function createEvaluationKnowledgeDraft(candidateId, payload) {
  return requestJson(`/api/declaration-qa/evaluation/candidates/${encodeURIComponent(candidateId)}/knowledge-draft`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function reviewEvaluationKnowledgeDraft(candidateId, payload) {
  return requestJson(`/api/declaration-qa/evaluation/candidates/${encodeURIComponent(candidateId)}/knowledge-draft/review`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function publishEvaluationKnowledgeDraft(candidateId) {
  return requestJson(`/api/declaration-qa/evaluation/candidates/${encodeURIComponent(candidateId)}/knowledge-draft/publish`, {
    method: 'POST',
  });
}

export function toolTraceStatus(ok, status) {
  if (ok) return 'completed';
  if (status === 'not_found') return 'empty';
  if (status === 'human_rejected') return 'rejected';
  if (status === 'pending') return 'running';
  return 'failed';
}

export function reduceToolTrace(current, event, data = {}) {
  const trace = current ? { ...current } : createToolTrace();
  trace.tools = [...(trace.tools || [])];
  trace.loaded = true;

  if (event === 'agent_decision') {
    trace.phase = data.action === 'tools' ? 'executing' : 'answering';
    for (const call of data.tool_calls || []) {
      const callId = String(call.id || call.call_id || '');
      const existingIndex = trace.tools.findIndex((item) => callId && item.callId === callId);
      const tool = {
        callId,
        toolName: String(call.name || call.tool_name || 'unknown_tool'),
        arguments: call.arguments || {},
        iteration: Number(data.iteration || 0),
        status: 'running',
        ok: null,
        latencyMs: null,
        parallel: false,
      };
      if (existingIndex >= 0) trace.tools[existingIndex] = { ...trace.tools[existingIndex], ...tool };
      else trace.tools.push(tool);
    }
  } else if (event === 'tool_executed') {
    trace.phase = 'checking';
    const callId = String(data.call_id || '');
    let index = trace.tools.findIndex((item) => callId && item.callId === callId);
    if (index < 0) {
      for (let i = trace.tools.length - 1; i >= 0; i -= 1) {
        if (trace.tools[i].toolName === data.tool_name && trace.tools[i].status === 'running') {
          index = i;
          break;
        }
      }
    }
    const completed = {
      status: toolTraceStatus(data.ok, data.status),
      ok: Boolean(data.ok),
      latencyMs: numberOrNull(data.latency_ms),
      parallel: Boolean(data.parallel),
    };
    if (index >= 0) {
      trace.tools[index] = {
        ...trace.tools[index],
        ...completed,
        callId: callId || trace.tools[index].callId,
        toolName: String(data.tool_name || trace.tools[index].toolName || 'unknown_tool'),
      };
    } else {
      trace.tools.push({
        ...completed,
        callId,
        toolName: String(data.tool_name || 'unknown_tool'),
        arguments: {},
        iteration: 0,
      });
    }
  } else if (event === 'human_confirmation_required') {
    trace.status = 'paused';
    trace.phase = 'awaiting_confirmation';
    const callIds = new Set((data.actions || []).map((item) => String(item.call_id || '')));
    trace.tools = trace.tools.map((item) => (
      callIds.has(String(item.callId || ''))
        ? { ...item, status: 'awaiting_confirmation', ok: null }
        : item
    ));
  } else if (event === 'human_confirmation_resolved') {
    trace.status = 'running';
    trace.phase = 'executing';
    const callIds = new Set((data.call_ids || []).map(String));
    trace.tools = trace.tools.map((item) => (
      callIds.has(String(item.callId || ''))
        ? { ...item, status: data.approved ? 'running' : 'rejected', ok: data.approved ? null : false }
        : item
    ));
  } else if (event === 'agent_observation') {
    trace.phase = 'checking';
    trace.observation = {
      toolResultCount: Number(data.tool_result_count || 0),
      knowledgeSourceCount: Number(data.knowledge_source_count || 0),
      selectedItemCode: String(data.selected_item_code || ''),
      selectedApplicationId: data.selected_application_id ?? null,
    };
  } else if (event === 'answer_generated') {
    trace.phase = 'answering';
  } else if (event === 'done') {
    trace.status = 'completed';
    trace.phase = 'completed';
    trace.runId = String(data.run_id || data.audit_id || trace.runId || '');
  } else if (event === 'error') {
    trace.status = 'failed';
    trace.phase = 'failed';
    trace.error = String(data.message || data.error || '请求失败');
    trace.tools = trace.tools.map((item) => item.status === 'running' ? { ...item, status: 'failed', ok: false } : item);
  }
  return trace;
}

export function traceFromAgentRun(run) {
  const iterationByCall = new Map();
  for (const message of run?.messages || []) {
    if (message?.message_type !== 'tool_call') continue;
    const iteration = Number(message.content_json?.iteration || 0);
    for (const call of message.content_json?.tool_calls || []) {
      const id = String(call.id || call.call_id || '');
      if (id) iterationByCall.set(id, iteration);
    }
  }

  const tools = (run?.tool_calls || []).map((call) => {
    const externalId = String(call.external_call_id || call.tool_call_id || '');
    return {
      callId: externalId,
      toolName: String(call.tool_name || 'unknown_tool'),
      arguments: call.arguments || {},
      iteration: iterationByCall.get(externalId) || 0,
      status: toolTraceStatus(call.ok, call.status),
      ok: Boolean(call.ok),
      latencyMs: numberOrNull(call.latency_ms),
      parallel: Boolean(call.metadata?.parallel),
    };
  });
  return {
    ...createToolTrace(run?.status === 'failed' ? 'failed' : 'completed'),
    phase: run?.status === 'failed' ? 'failed' : 'completed',
    loaded: true,
    runId: String(run?.run_id || ''),
    tools,
    error: String(run?.error_message || ''),
  };
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
