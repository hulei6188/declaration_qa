import React from 'react';
import {
  FlaskConical,
  Headset,
  GitBranch,
  Landmark,
  LibraryBig,
  Moon,
  PanelLeftClose,
  RefreshCw,
  SquarePen,
  Sun,
  Trash2,
} from 'lucide-react';
import { escapeHtml } from '../utils.js';

const NAV_ITEMS = [
  { id: 'human-service', label: '人工服务', icon: Headset },
  { id: 'library', label: '文件库', icon: LibraryBig },
  { id: 'versions', label: '版本管理', icon: GitBranch },
  { id: 'evaluation', label: '评测中心', icon: FlaskConical },
];

export default function Sidebar({
  activeView,
  sessions,
  sessionId,
  theme,
  onNewChat,
  onViewChange,
  onRefreshSessions,
  onLoadSession,
  onDeleteSession,
  onToggleTheme,
  onCollapse,
}) {
  return (
    <aside className="context-pane app-sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-mark"><Landmark size={18} /></span>
        <strong>我的agent</strong>
        <button className="icon-button" onClick={onCollapse} title="折叠侧栏" aria-label="折叠侧栏">
          <PanelLeftClose size={16} />
        </button>
      </div>

      <nav className="sidebar-primary-nav" aria-label="主导航">
        <button className={activeView === 'chat' ? 'active' : ''} onClick={onNewChat}>
          <SquarePen size={17} />
          <span>新对话</span>
        </button>
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button key={id} className={activeView === id ? 'active' : ''} onClick={() => onViewChange(id)}>
            <Icon size={17} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <section className="sidebar-recents">
        <div className="sidebar-section-title">
          <span>最近对话</span>
          <button className="icon-button" onClick={onRefreshSessions} title="刷新会话" aria-label="刷新会话">
            <RefreshCw size={14} />
          </button>
        </div>
        <div className="sidebar-session-list">
          {sessions.length === 0 ? (
            <p className="sidebar-empty">暂无会话</p>
          ) : sessions.slice(0, 14).map((session) => (
            <div key={session.session_id} className={`sidebar-session${session.session_id === sessionId ? ' active' : ''}`}>
              <button className="sidebar-session-open" onClick={() => onLoadSession(session.session_id)}>
                <span>{escapeHtml(session.title || '政务服务咨询')}</span>
              </button>
              <button
                className="sidebar-session-delete"
                onClick={() => onDeleteSession(session.session_id)}
                title="删除会话"
                aria-label={`删除${session.title || '会话'}`}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </section>

      <div className="sidebar-footer">
        <button onClick={onToggleTheme}>
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          <span>{theme === 'dark' ? '浅色模式' : '深色模式'}</span>
        </button>
        <div className="sidebar-profile">
          <span>政</span>
          <div><strong>政务服务工作台</strong><small>Production</small></div>
        </div>
      </div>
    </aside>
  );
}
