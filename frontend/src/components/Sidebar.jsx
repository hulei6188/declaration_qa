import React, { useState } from 'react';
import { RefreshCw, Trash2, Upload } from 'lucide-react';
import { documentRevision, escapeHtml, etlStageLabel, formatBytes } from '../utils.js';

export default function Sidebar({
  sessions, documents, selectedDocId, uploadItems, uploadResult, uploadInFlight,
  docManageResult, onRefreshSessions, onLoadSession, onUpload, onSelectDocument,
  onDeleteSession, onRefreshDocuments, onSetDocumentStatus, onUpdateRollout, onCreateVersion,
  onCompareVersion, onRollback,
}) {
  const [docTitle, setDocTitle] = useState('');
  const [docContent, setDocContent] = useState('');
  const [docFile, setDocFile] = useState(null);
  const [chunkingStrategy, setChunkingStrategy] = useState('chonkie');
  const [policyVersion, setPolicyVersion] = useState('2025');
  const [versionContent, setVersionContent] = useState('');

  function handleUpload() {
    const files = docFile ? Array.from(docFile) : [];
    onUpload(docTitle, docContent, files, chunkingStrategy, policyVersion);
  }

  function handleFileChange(e) {
    const files = e.target.files;
    setDocFile(files?.length ? files : null);
  }

  function handleDocumentKeyDown(e, docId) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    onSelectDocument(docId);
  }

  return (
    <aside className="context-pane">
      {/* Sessions */}
      <section className="panel panel-compact session-panel">
        <div className="panel-title">
          <span>会话历史</span>
          <button className="icon-button" onClick={onRefreshSessions} title="刷新">
            <RefreshCw size={14} />
          </button>
        </div>
        <div className="session-list">
          {sessions.length === 0 ? (
            <p className="hint">暂无会话</p>
          ) : (
            sessions.slice(0, 8).map((s) => (
              <div key={s.session_id} className="session-item">
                <button className="session-open" onClick={() => onLoadSession(s.session_id)} title="打开会话">
                  <strong>{escapeHtml(s.title || '申报智能问答')}</strong>
                  <code>{escapeHtml(s.biz_type || '-')} · {escapeHtml(s.region_code || '-')} · {s.message_count || 0} 条</code>
                </button>
                <button className="session-delete" onClick={() => onDeleteSession(s.session_id)} title="删除会话">
                  <Trash2 size={13} />
                </button>
              </div>
            ))
          )}
        </div>
      </section>

      {/* Version Management */}
      <section className="panel version-panel">
        <div className="panel-title">
          <span>版本管理</span>
          <span className="count-badge">{documents.length} 个</span>
          <button className="icon-button" onClick={onRefreshDocuments} title="刷新">
            <RefreshCw size={14} />
          </button>
        </div>
        <div className="document-list">
          {documents.length === 0 ? (
            <p className="hint">暂无知识文档</p>
          ) : (
            documents.map((doc) => {
              const revision = documentRevision(doc);
              return (
                <div
                  key={doc.document_id}
                  role="button"
                  tabIndex={0}
                  className={`document-item ${doc.document_id === selectedDocId ? 'active' : ''}`}
                  onClick={() => onSelectDocument(doc.document_id)}
                  onKeyDown={(e) => handleDocumentKeyDown(e, doc.document_id)}
                >
                  <strong>{escapeHtml(doc.title || '未命名文档')}</strong>
                  <div className="doc-facts">
                    <span className={`doc-fact doc-status status-${statusClass(revision.status)}`}>{etlStageLabel(revision.status)}</span>
                    <span className="doc-fact">v{doc.version_no}</span>
                    <span className="doc-fact">{doc.rollout_percent}%</span>
                    <span className="doc-fact">{escapeHtml(doc.policy_version || '-')}</span>
                    <span className="doc-fact">{doc.chunk_count || 0} 块</span>
                  </div>
                  <div className="revision-strip">
                    <span>{etlStageLabel(revision.stage)}</span>
                    <em>{escapeHtml(revision.revision)}</em>
                  </div>
                </div>
              );
            })
          )}
        </div>
        <div className="field-grid">
          <button onClick={() => onSetDocumentStatus('publish')}>发布</button>
          <button onClick={() => onSetDocumentStatus('unpublish')}>下线</button>
        </div>
        <div className="field-grid align-end">
          <label>
            灰度比例
            <input id="rolloutPercent" type="number" min="0" max="100" defaultValue="100" />
          </label>
          <button onClick={onUpdateRollout}>更新灰度</button>
        </div>
        <label>
          新版本内容
          <textarea rows="3" value={versionContent} onChange={(e) => setVersionContent(e.target.value)} placeholder="基于选中文档创建 draft 版本" />
        </label>
        <div className="field-grid">
          <button onClick={() => { onCreateVersion(versionContent); setVersionContent(''); }}>创建版本</button>
          <button onClick={onCompareVersion}>对比版本</button>
        </div>
        <button className="ghost full" onClick={onRollback}>回滚到选中版本</button>
        <p className="hint">{docManageResult}</p>
      </section>

      {/* Upload */}
      <section className="panel upload-panel">
        <div className="panel-title">
          <span>知识入库</span>
        </div>
        <label>
          标题（可选）
          <input value={docTitle} onChange={(e) => setDocTitle(e.target.value)} placeholder="文件入库默认使用文件名；文本入库可自定义标题" />
        </label>
        <label>
          选择文件（可批量）
          <input type="file" multiple onChange={handleFileChange} accept=".txt,.md,.csv,.pdf,.docx,.xlsx,.png,.jpg,.jpeg,.webp" />
        </label>
        <label>
          切分方法
          <select id="chunkingStrategy" value={chunkingStrategy} onChange={(e) => setChunkingStrategy(e.target.value)}>
            <option value="chonkie">Chonkie 结构分块</option>
            <option value="parent_child">父子分块</option>
            <option value="hierarchical">层级分块</option>
          </select>
        </label>
        <div className="upload-queue">
          {uploadItems.length === 0 ? (
            <p className="hint">可一次选择多个文件批量入库。</p>
          ) : (
            uploadItems.map((item) => (
              <div key={item.id} className={`upload-queue-item ${item.status}`}>
                <span className="upload-state-dot" />
                <div>
                  <strong>{escapeHtml(item.name)}</strong>
                  <code>{formatBytes(item.size)} · {escapeHtml(item.strategyLabel || '')} · {escapeHtml(item.message || '')}{item.chunksIndexed != null ? ` · ${item.chunksIndexed} 分块` : ''}</code>
                  <EtlProgress item={item} />
                </div>
              </div>
            ))
          )}
        </div>
        <label>
          文档内容（不选文件时使用）
          <textarea rows="5" value={docContent} onChange={(e) => setDocContent(e.target.value)} placeholder="选择文件入库时不用填写；只做文本入库时粘贴内容" />
        </label>
        <label>
          政策版本
          <input value={policyVersion} onChange={(e) => setPolicyVersion(e.target.value)} />
        </label>
        <button className="primary full" onClick={handleUpload} disabled={uploadInFlight}>
          <Upload size={15} />
          上传并索引
        </button>
        <p className="hint">{uploadResult}</p>
      </section>
    </aside>
  );
}

function EtlProgress({ item }) {
  const etl = item.etl || {};
  const stages = Array.isArray(etl.stages) ? etl.stages : [];
  const current = etl.current_stage || '';
  const progress = Math.max(0, Math.min(100, Number(item.progress || 0)));
  const visibleStages = stages.slice(-5);
  if (!current && visibleStages.length === 0) return null;
  return (
    <div className="etl-progress">
      <div className="etl-bar" aria-hidden="true">
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="etl-stage-row">
        <span>{etlStageLabel(current)}</span>
        <span>{progress}%</span>
      </div>
      {visibleStages.length > 0 && (
        <div className="etl-stage-list">
          {visibleStages.map((stage, index) => (
            <span key={`${stage.stage}-${index}`} className={stage.stage === current ? 'active' : ''}>
              {etlStageLabel(stage.stage)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function statusClass(status) {
  return String(status || 'unknown').toLowerCase().replace(/[^a-z0-9_-]/g, '-');
}
