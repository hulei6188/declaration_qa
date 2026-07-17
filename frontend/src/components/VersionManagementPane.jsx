import React, { useEffect, useMemo, useState } from 'react';
import {
  ArchiveRestore,
  CheckCircle2,
  GitCompare,
  PanelLeftOpen,
  RefreshCw,
  Rocket,
  Search,
  UploadCloud,
} from 'lucide-react';
import { documentRevision, escapeHtml, etlStageLabel } from '../utils.js';

export default function VersionManagementPane({
  documents,
  selectedDocId,
  docManageResult,
  onSelectDocument,
  onRefreshDocuments,
  onSetDocumentStatus,
  onUpdateRollout,
  onCreateVersion,
  onCompareVersion,
  onRollback,
  sidebarCollapsed,
  onToggleSidebar,
}) {
  const [query, setQuery] = useState('');
  const [versionContent, setVersionContent] = useState('');
  const [rolloutPercent, setRolloutPercent] = useState(100);
  const selected = documents.find((doc) => doc.document_id === selectedDocId) || null;
  const filtered = useMemo(() => documents.filter((doc) => !query.trim() || String(doc.title || '').toLowerCase().includes(query.trim().toLowerCase())), [documents, query]);
  const revision = selected ? documentRevision(selected) : null;

  useEffect(() => {
    setRolloutPercent(Number(selected?.rollout_percent ?? 100));
  }, [selected?.document_id, selected?.rollout_percent]);

  return (
    <section className="management-pane version-management-pane">
      <header className="management-header">
        <div className="management-title">
          {sidebarCollapsed && <button className="icon-button" onClick={onToggleSidebar} title="展开侧栏" aria-label="展开侧栏"><PanelLeftOpen size={17} /></button>}
          <div><span>发布控制</span><h1>版本管理</h1></div>
        </div>
        <div className="management-header-actions">
          <label className="management-search"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文档" /></label>
          <button className="icon-button" onClick={onRefreshDocuments} title="刷新版本" aria-label="刷新版本"><RefreshCw size={16} /></button>
        </div>
      </header>

      <div className="version-workspace">
        <aside className="version-document-list">
          <div className="version-list-heading"><span>文档</span><em>{filtered.length}</em></div>
          {filtered.map((doc) => {
            const itemRevision = documentRevision(doc);
            return (
              <button key={doc.document_id} className={doc.document_id === selectedDocId ? 'active' : ''} onClick={() => onSelectDocument(doc.document_id)}>
                <strong>{escapeHtml(doc.title || '未命名文档')}</strong>
                <span><em className={`status-label status-${statusClass(itemRevision.status)}`}>{etlStageLabel(itemRevision.status)}</em><small>v{doc.version_no}</small><small>{doc.rollout_percent}%</small></span>
              </button>
            );
          })}
        </aside>

        <main className="version-editor">
          {!selected ? (
            <div className="management-empty"><GitCompare size={30} /><strong>选择一个文档管理版本</strong></div>
          ) : (
            <>
              <div className="version-editor-title">
                <div><span>当前文档</span><h2>{escapeHtml(selected.title || '未命名文档')}</h2><p>{escapeHtml(revision.revision)} · v{selected.version_no} · {selected.chunk_count || 0} 分块</p></div>
                <em className={`status-label large status-${statusClass(revision.status)}`}>{etlStageLabel(revision.status)}</em>
              </div>

              <section className="version-operation-band">
                <div><span>发布状态</span><strong>{revision.status === 'published' ? '线上生效' : '未发布'}</strong></div>
                <div className="version-actions"><button onClick={() => onSetDocumentStatus('publish')}><Rocket size={15} />发布</button><button onClick={() => onSetDocumentStatus('unpublish')}>下线</button></div>
              </section>

              <section className="version-operation-band rollout-band">
                <div><span>灰度比例</span><strong>{rolloutPercent}%</strong></div>
                <div className="rollout-control"><input id="rolloutPercent" type="range" min="0" max="100" value={rolloutPercent} onChange={(event) => setRolloutPercent(Number(event.target.value))} /><button onClick={onUpdateRollout}>应用灰度</button></div>
              </section>

              <section className="version-create-area">
                <div className="version-section-heading"><div><span>新版本</span><h2>创建 Draft</h2></div><UploadCloud size={18} /></div>
                <textarea rows="9" value={versionContent} onChange={(event) => setVersionContent(event.target.value)} placeholder="输入新版本内容" />
                <div className="version-editor-actions">
                  <button className="primary" onClick={() => { onCreateVersion(versionContent); setVersionContent(''); }} disabled={!versionContent.trim()}><UploadCloud size={15} />创建版本</button>
                  <button onClick={onCompareVersion}><GitCompare size={15} />对比版本</button>
                  <button onClick={onRollback}><ArchiveRestore size={15} />回滚</button>
                </div>
              </section>
              <div className="version-result"><CheckCircle2 size={15} /><span>{docManageResult}</span></div>
            </>
          )}
        </main>
      </div>
    </section>
  );
}

function statusClass(status) {
  return String(status || 'unknown').toLowerCase().replace(/[^a-z0-9_-]/g, '-');
}
