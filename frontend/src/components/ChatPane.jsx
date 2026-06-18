import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2 } from 'lucide-react';
import { escapeHtml, ticketLabel } from '../utils.js';

export default function ChatPane({
  messages, streaming, sendState, currentTicket, ticketPickerOpen, ticketList,
  pendingTicketQuery, onSend, onClear, onOpenTicketPicker, onCloseTicketPicker,
  onSelectTicket, onClearTicket, ctx, setCtx,
}) {
  const [input, setInput] = useState('');
  const [autoFollow, setAutoFollow] = useState(true);
  const messagesRef = useRef(null);
  const messagesEnd = useRef(null);
  const previousCount = useRef(messages.length);

  useEffect(() => {
    const hasNewMessage = messages.length > previousCount.current;
    previousCount.current = messages.length;
    if (hasNewMessage) {
      setAutoFollow(true);
    }
    if (!autoFollow && !hasNewMessage) return;
    messagesEnd.current?.scrollIntoView({ behavior: streaming ? 'auto' : 'smooth' });
  }, [messages, autoFollow, streaming]);

  function handleMessagesScroll(e) {
    const el = e.currentTarget;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAutoFollow(distanceFromBottom < 48);
  }

  function handleSubmit(e) {
    e.preventDefault();
    const q = input.trim();
    if (!q || streaming) return;
    setAutoFollow(true);
    onSend(q);
    setInput('');
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  const ticketId = currentTicket?.ticket_id || ctx.applyId;
  const ticketDisplay = ticketId ? ticketLabel(currentTicket || { ticket_id: ticketId }) : null;

  return (
    <section className="chat-pane">
      {/* Header */}
      <div className="chat-header">
        <div>
          <p className="eyebrow">Live SSE Chat</p>
          <h2>问答会话</h2>
          <div className="current-ticket">
            <span>{ticketDisplay ? `当前工单：${ticketDisplay}` : '当前工单：未选择'}</span>
            <button className="ghost compact" onClick={onOpenTicketPicker}>更换</button>
            <button className="ghost compact" onClick={onClearTicket} disabled={!ticketId}>清除</button>
          </div>
          {ticketPickerOpen && (
            <div className="ticket-picker">
              <strong>选择当前工单</strong>
              {ticketList.length === 0 ? (
                <p className="hint">暂无可选择工单。</p>
              ) : (
                ticketList.map((t) => (
                  <button key={t.ticket_id} className="ticket-option" onClick={() => onSelectTicket(t)}>
                    <span>{escapeHtml(t.title || t.biz_type || '申报工单')}</span>
                    <code>{escapeHtml(t.ticket_id || '-')} · {escapeHtml(t.node_name || t.current_node || t.status || '-')} · {escapeHtml(t.updated_at || '-')}</code>
                  </button>
                ))
              )}
              <button className="ghost compact" onClick={onCloseTicketPicker}>取消</button>
            </div>
          )}
        </div>
        <button className="ghost" onClick={onClear}>
          <Trash2 size={15} />
          清空
        </button>
      </div>

      {/* Messages */}
      <div className="messages" ref={messagesRef} onScroll={handleMessagesScroll}>
        {messages.map((m, i) => (
          <article key={i} className={`message ${m.role}`}>
            <span className="message-meta">{m.meta}</span>
            <div className="message-content">{m.content}</div>
            {m.ticketSelection && (
              <div className="ticket-selection">
                <strong>{m.ticketSelection.reason === 'multiple_candidates' ? '请选择对应工单' : '候选工单'}</strong>
                {m.ticketSelection.tickets.map((t) => (
                  <button key={t.ticket_id} className="ticket-option" onClick={() => onSelectTicket(t)}>
                    <span>{escapeHtml(t.title || t.biz_type || '申报工单')}</span>
                    <code>{escapeHtml(t.ticket_id || '-')} · {escapeHtml(t.node_name || t.current_node || t.status || '-')} · {escapeHtml(t.updated_at || '-')}</code>
                  </button>
                ))}
              </div>
            )}
          </article>
        ))}
        <div ref={messagesEnd} />
      </div>

      {/* Composer */}
      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          rows="3"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入问题，例如：办理居住证需要哪些材料？"
          disabled={streaming}
        />
        <div className="composer-actions">
          <span>{sendState}</span>
          <button className="primary" type="submit" disabled={streaming || !input.trim()}>
            <Send size={15} />
            发送
          </button>
        </div>
      </form>
    </section>
  );
}
