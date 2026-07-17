import React, { useMemo, useRef, useState } from 'react';
import {
  Check,
  File,
  FileSearch,
  FolderOpen,
  Image,
  PanelLeftOpen,
  RefreshCw,
  Search,
  Upload,
} from 'lucide-react';
import { documentRevision, escapeHtml, etlStageLabel, formatBytes } from '../utils.js';

export default function KnowledgeLibraryPane({
  documents,
  selectedDocId,
  uploadItems,
  uploadResult,
  uploadInFlight,
  onUpload,
  onSelectDocument,
  onRefreshDocuments,
  sidebarCollapsed,
  onToggleSidebar,
}) {
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [docTitle, setDocTitle] = useState('');
  const [docContent, setDocContent] = useState('');
  const [files, setFiles] = useState([]);
  const [chunkingStrategy, setChunkingStrategy] = useState('chonkie');
  const [policyVersion, setPolicyVersion] = useState('2025');
  const fileInputRef = useRef(null);

  const filteredDocuments = useMemo(() => documents.filter((doc) => {
    const revision = documentRevision(doc);
    const matchesQuery = !query.trim() || String(doc.title || '').toLowerCase().includes(query.trim().toLowerCase());
    const matchesFilter = filter === 'all'
      || (filter === 'published' && revision.status === 'published')
      || (filter === 'processing' && revision.status !== 'published');
    return matchesQuery && matchesFilter;
  }), [documents, filter, query]);

  function handleUpload() {
    onUpload(docTitle, docContent, files, chunkingStrategy, policyVersion);
  }

  return (
    <section className="management-pane library-pane">
      <header className="management-header">
        <div className="management-title">
          {sidebarCollapsed && (
            <button className="icon-button" onClick={onToggleSidebar} title="展开侧栏" aria-label="展开侧栏">
              <PanelLeftOpen size={17} />
            </button>
          )}
          <div><span>知识资产</span><h1>文件库</h1></div>
        </div>
        <div className="management-header-actions">
          <label className="management-search">
            <Search size={15} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文件" />
          </label>
          <button className="icon-button" onClick={onRefreshDocuments} title="刷新文件" aria-label="刷新文件"><RefreshCw size={16} /></button>
          <button className="primary" onClick={() => fileInputRef.current?.click()}><Upload size={15} />上传</button>
        </div>
      </header>

      <div className="library-filter-bar">
        <div className="segmented-control" aria-label="文件筛选">
          <button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>全部</button>
          <button className={filter === 'published' ? 'active' : ''} onClick={() => setFilter('published')}>已发布</button>
          <button className={filter === 'processing' ? 'active' : ''} onClick={() => setFilter('processing')}>处理中</button>
        </div>
        <span>{filteredDocuments.length} 个文件</span>
      </div>

      <div className="library-workspace">
        <main className="library-table-wrap">
          <div className="library-table-head">
            <span>名称</span><span>状态</span><span>版本</span><span>政策版本</span><span>分块</span>
          </div>
          <div className="library-table-body">
            {filteredDocuments.map((doc) => {
              const revision = documentRevision(doc);
              const selected = doc.document_id === selectedDocId;
              const isImage = /\.(png|jpe?g|webp|gif)$/i.test(doc.title || '');
              const Icon = isImage ? Image : File;
              return (
                <button
                  key={doc.document_id}
                  className={`library-row${selected ? ' selected' : ''}`}
                  onClick={() => onSelectDocument(doc.document_id)}
                >
                  <span className="library-file-name"><span className="file-type-icon"><Icon size={17} /></span><strong>{escapeHtml(doc.title || '未命名文件')}</strong></span>
                  <span><em className={`status-label status-${statusClass(revision.status)}`}>{etlStageLabel(revision.status)}</em></span>
                  <span>v{doc.version_no}</span>
                  <span>{escapeHtml(doc.policy_version || '-')}</span>
                  <span>{doc.chunk_count || 0}</span>
                  {selected && <Check className="library-row-check" size={15} />}
                </button>
              );
            })}
            {!filteredDocuments.length && (
              <div className="management-empty"><FolderOpen size={28} /><strong>没有匹配的文件</strong></div>
            )}
          </div>
        </main>

        <aside className="upload-workbench">
          <div className="workbench-heading"><div><span>知识入库</span><h2>上传与索引</h2></div><Upload size={18} /></div>
          <label>标题（可选）<input value={docTitle} onChange={(event) => setDocTitle(event.target.value)} placeholder="默认使用文件名" /></label>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            hidden
            accept=".txt,.md,.csv,.pdf,.docx,.xlsx,.png,.jpg,.jpeg,.webp"
            onChange={(event) => setFiles(Array.from(event.target.files || []))}
          />
          <button className="file-picker" onClick={() => fileInputRef.current?.click()}>
            <FileSearch size={19} />
            <span>{files.length ? `已选择 ${files.length} 个文件` : '选择一个或多个文件'}</span>
          </button>
          {files.length > 0 && <div className="selected-files">{files.slice(0, 4).map((file) => <span key={`${file.name}-${file.size}`}>{file.name}<small>{formatBytes(file.size)}</small></span>)}</div>}
          <label>切分方法<select id="chunkingStrategy" value={chunkingStrategy} onChange={(event) => setChunkingStrategy(event.target.value)}><option value="chonkie">Chonkie 结构分块</option><option value="parent_child">父子分块</option><option value="hierarchical">层级分块</option></select></label>
          <label>政策版本<input value={policyVersion} onChange={(event) => setPolicyVersion(event.target.value)} /></label>
          <label>文本内容<textarea rows="6" value={docContent} onChange={(event) => setDocContent(event.target.value)} placeholder="不上传文件时，可直接粘贴文本" /></label>
          <button className="primary full" onClick={handleUpload} disabled={uploadInFlight}><Upload size={15} />上传并索引</button>
          <p className="workbench-status">{uploadResult}</p>
          <div className="upload-queue compact-upload-queue">
            {uploadItems.map((item) => <UploadQueueItem key={item.id} item={item} />)}
          </div>
        </aside>
      </div>
    </section>
  );
}

function UploadQueueItem({ item }) {
  const progress = Math.max(0, Math.min(100, Number(item.progress || 0)));
  return (
    <div className={`upload-queue-item ${item.status}`}>
      <span className="upload-state-dot" />
      <div><strong>{escapeHtml(item.name)}</strong><code>{escapeHtml(item.message || '')}{item.chunksIndexed != null ? ` · ${item.chunksIndexed} 分块` : ''}</code><div className="etl-bar"><span style={{ width: `${progress}%` }} /></div></div>
    </div>
  );
}

function statusClass(status) {
  return String(status || 'unknown').toLowerCase().replace(/[^a-z0-9_-]/g, '-');
}
