// ---- Constants ----
export const API_BASE = '';
export const INDEX_JOB_MAX_ATTEMPTS = 180;
export const CURRENT_TICKET_STORAGE_KEY = 'declarationQaCurrentTicket';

export const DEFAULT_CONTEXT = {
  userId: 1,
  regionCode: '340000',
  bizType: '',
  appVersion: '2.0.0',
  policyVersion: '2025',
  applyId: '',
};

// ---- Formatters ----
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

export function ticketLabel(ticket) {
  return [
    ticket.title || ticket.biz_type || '申报工单',
    ticket.ticket_id,
    ticket.node_name || ticket.current_node || ticket.status,
  ].filter(Boolean).join(' / ');
}

// ---- Persistence ----
export function loadTicket() {
  try {
    const saved = JSON.parse(localStorage.getItem(CURRENT_TICKET_STORAGE_KEY) || 'null');
    return saved?.ticket?.ticket_id ? saved : null;
  } catch {
    return null;
  }
}

export function saveTicket(ticket, ctx) {
  if (!ticket?.ticket_id) return;
  localStorage.setItem(CURRENT_TICKET_STORAGE_KEY, JSON.stringify({ ticket, requestContext: ctx }));
}

export function clearTicket() {
  localStorage.removeItem(CURRENT_TICKET_STORAGE_KEY);
}

// ---- API ----
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

// Health
export function checkHealth() {
  return requestJson('/api/declaration-qa/health');
}

// Chat SSE
export function buildChatRequest(query, sessionId, ctx) {
  const payload = {
    query,
    user_id: Number(ctx.userId || 1),
    region_code: ctx.regionCode || '340000',
    app_version: ctx.appVersion,
    policy_version: ctx.policyVersion || '2025',
    session_id: sessionId,
  };
  if (ctx.bizType) payload.biz_type = ctx.bizType;
  if (ctx.applyId) payload.apply_id = ctx.applyId;
  return payload;
}

export async function startChatStream(body) {
  return fetch(`${API_BASE}/api/declaration-qa/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// Sessions
export function fetchSessions(userId) {
  return requestJson(`/api/declaration-qa/sessions?user_id=${encodeURIComponent(userId || 1)}`);
}

export function fetchSession(sessionId, userId) {
  return requestJson(`/api/declaration-qa/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId || 1)}`);
}

// Tickets
export function fetchTickets(userId, regionCode, bizType) {
  return requestJson(
    `/api/declaration-qa/tickets?user_id=${encodeURIComponent(userId || '1')}&region_code=${encodeURIComponent(regionCode || '340000')}&biz_type=${encodeURIComponent(bizType || '')}`
  );
}

// Documents
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

// Index jobs
export function fetchIndexJob(jobId) {
  return requestJson(`/api/declaration-qa/index-jobs/${jobId}`);
}

// Feedback
export function sendFeedback(auditId, rating) {
  return request('/api/declaration-qa/feedback', {
    method: 'POST',
    body: JSON.stringify({ audit_id: auditId, rating, comment: '' }),
  });
}

// Evaluation
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

// SSE helpers
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
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) yield parseSSE(part);
  }
  if (buffer.trim()) yield parseSSE(buffer);
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
