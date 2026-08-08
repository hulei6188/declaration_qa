import React, { useEffect, useState } from 'react';
import { CheckCircle2, Headset, MessageSquare, PanelLeftOpen, RefreshCw, Send, UserRound } from 'lucide-react';
import {
  claimHumanServiceTicket,
  fetchHumanServiceTickets,
  sendHumanServiceMessage,
  updateHumanServiceTicketStatus,
} from '../utils.js';

const STATUS_LABELS = { open: '待接入', claimed: '处理中', waiting_user: '等待用户', resolved: '已解决', closed: '已关闭' };

export default function HumanServicePane({ sidebarCollapsed, onToggleSidebar }) {
  const [tickets, setTickets] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [reply, setReply] = useState('');
  const [message, setMessage] = useState('');

  async function load() {
    try {
      const next = await fetchHumanServiceTickets();
      setTickets(next || []);
      setSelectedId((current) => current || next?.[0]?.ticket_id || '');
    } catch (error) { setMessage(`工单读取失败：${error.message}`); }
  }
  useEffect(() => { load(); }, []);
  const selected = tickets.find((ticket) => ticket.ticket_id === selectedId) || null;

  async function refreshTicket(action) {
    try { await action(); await load(); } catch (error) { setMessage(`操作失败：${error.message}`); }
  }
  async function sendReply() {
    if (!selected || !reply.trim()) return;
    await refreshTicket(() => sendHumanServiceMessage(selected.ticket_id, reply.trim()));
    setReply('');
  }

  return (
    <section className="human-service-pane">
      <header className="chat-header evaluation-header">
        <div className="chat-heading">
          {sidebarCollapsed && <button className="icon-button" onClick={onToggleSidebar} title="展开侧栏" aria-label="展开侧栏"><PanelLeftOpen size={16} /></button>}
          <div><span className="page-kicker">人工服务</span><h2>坐席工作台</h2></div>
        </div>
        <button className="icon-button" onClick={load} title="刷新工单" aria-label="刷新工单"><RefreshCw size={15} /></button>
      </header>
      <div className="human-service-workspace">
        <aside className="human-service-ticket-list">
          <div className="version-list-heading"><span>待处理工单</span><em>{tickets.filter((ticket) => !['closed', 'resolved'].includes(ticket.status)).length}</em></div>
          {tickets.map((ticket) => (
            <button key={ticket.ticket_id} className={ticket.ticket_id === selectedId ? 'active' : ''} onClick={() => setSelectedId(ticket.ticket_id)}>
              <strong>{ticket.query || '人工服务请求'}</strong>
              <span><em>{STATUS_LABELS[ticket.status] || ticket.status}</em><small>{ticket.intent || '未识别意图'}</small></span>
            </button>
          ))}
          {!tickets.length && <p className="hint">暂无人工服务工单</p>}
        </aside>
        <main className="human-service-detail">
          {!selected ? <div className="management-empty"><Headset size={30} /><strong>选择一个工单查看上下文</strong></div> : (
            <>
              <div className="human-service-title"><div><span>工单 {selected.ticket_id}</span><h2>{selected.query || '人工服务请求'}</h2><p>{selected.intent || '未识别意图'} · {selected.route || '未识别路由'} · 用户 {selected.user_id}</p></div><em className={`status-label large status-${selected.status}`}>{STATUS_LABELS[selected.status] || selected.status}</em></div>
              <div className="human-service-actions">
                <button onClick={() => refreshTicket(() => claimHumanServiceTicket(selected.ticket_id))} disabled={selected.status !== 'open'}><UserRound size={14} />接入工单</button>
                <button onClick={() => refreshTicket(() => updateHumanServiceTicketStatus(selected.ticket_id, 'waiting_user'))}>等待用户</button>
                <button onClick={() => refreshTicket(() => updateHumanServiceTicketStatus(selected.ticket_id, 'resolved'))}><CheckCircle2 size={14} />标记解决</button>
              </div>
              <section className="human-service-context-grid">
                <ContextBlock title="意图与路由" value={{ intent: selected.context?.intent, sub_intent: selected.context?.sub_intent, route: selected.context?.route, biz_type: selected.context?.biz_type }} />
                <ContextBlock title="已提取信息" value={selected.context?.dialog_task?.values || selected.context?.business_snapshot || {}} />
                <ContextBlock title="匹配证据" value={(selected.context?.retrieved_sources || []).slice(0, 4).map((source) => ({ title: source.title, doc_id: source.doc_id, snippet: source.snippet || source.evidence_text }))} />
                <ContextBlock title="最近工具结果" value={(selected.context?.tool_results || []).slice(-6).map((result) => ({ tool: result.tool_name, status: result.status, ok: result.ok }))} />
              </section>
              <section className="human-service-conversation">
                <div className="version-section-heading"><div><span>会话接管</span><h2>人工回复</h2></div><MessageSquare size={18} /></div>
                <div className="human-service-messages">{(selected.messages || []).map((item, index) => <div key={`${item.created_at}-${index}`} className={`human-service-message ${item.role}`}><strong>{item.role === 'agent' ? '坐席' : '用户'}</strong><span>{item.content}</span></div>)}</div>
                <div className="human-service-reply"><textarea rows="3" value={reply} onChange={(event) => setReply(event.target.value)} placeholder="输入给用户的回复" /><button className="primary" onClick={sendReply} disabled={!reply.trim()}><Send size={14} />发送</button></div>
              </section>
            </>
          )}
        </main>
      </div>
      {message && <div className="evaluation-status">{message}</div>}
    </section>
  );
}

function ContextBlock({ title, value }) {
  return <article className="human-service-context-block"><span>{title}</span><pre>{JSON.stringify(value || {}, null, 2)}</pre></article>;
}
