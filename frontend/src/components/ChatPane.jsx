import React, { useState, useRef, useEffect } from 'react';
import { ImagePlus, Maximize2, Minimize2, Send, Trash2, X } from 'lucide-react';
import { escapeHtml, ticketLabel } from '../utils.js';

export default function ChatPane({
  messages, streaming, sendState, currentTicket, ticketPickerOpen, ticketList,
  pendingTicketQuery, onSend, onClear, onOpenTicketPicker, onCloseTicketPicker,
  onSelectTicket, onClearTicket, ctx, setCtx, expanded = false, onToggleExpanded,
}) {
  const [input, setInput] = useState('');
  const [imageDraft, setImageDraft] = useState(null);
  const [autoFollow, setAutoFollow] = useState(true);
  const messagesRef = useRef(null);
  const messagesEnd = useRef(null);
  const previousCount = useRef(messages.length);
  const imageInputRef = useRef(null);

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
    if ((!q && !imageDraft) || streaming) return;
    setAutoFollow(true);
    const query = q || '请根据这张图片检索相关申报材料';
    onSend(query, undefined, undefined, undefined, imageDraft ? { ...imageDraft, text_present: Boolean(q) } : null);
    setInput('');
    setImageDraft(null);
    if (imageInputRef.current) imageInputRef.current.value = '';
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  function handleImageChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      e.target.value = '';
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setImageDraft({
        filename: file.name,
        content_type: file.type || 'image/png',
        size_bytes: file.size,
        data_url: String(reader.result || ''),
      });
    };
    reader.readAsDataURL(file);
  }

  function clearImageDraft() {
    setImageDraft(null);
    if (imageInputRef.current) imageInputRef.current.value = '';
  }

  const ticketId = currentTicket?.ticket_id || ctx.applyId;
  const ticketDisplay = ticketId ? ticketLabel(currentTicket || { ticket_id: ticketId }) : null;

  return (
    <section className={`chat-pane${expanded ? ' is-expanded' : ''}`}>
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
        <div className="chat-actions">
          <button
            type="button"
            className="icon-button"
            onClick={onToggleExpanded}
            title={expanded ? '还原会话窗口' : '放大会话窗口'}
            aria-label={expanded ? '还原会话窗口' : '放大会话窗口'}
          >
            {expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
          <button type="button" className="ghost" onClick={onClear}>
            <Trash2 size={15} />
            清空
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="messages" ref={messagesRef} onScroll={handleMessagesScroll}>
        {messages.map((m, i) => (
          <article key={i} className={`message ${m.role}`}>
            <span className="message-meta">{m.meta}</span>
            <div className="message-content">{m.content}</div>
            {m.attachments?.length > 0 && (
              <div className="message-attachments">
                {m.attachments.map((item, index) => (
                  <img key={`${item.filename || 'image'}-${index}`} src={item.data_url} alt={item.filename || '上传图片'} />
                ))}
              </div>
            )}
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
        {imageDraft && (
          <div className="composer-image">
            <img src={imageDraft.data_url} alt={imageDraft.filename || '待发送图片'} />
            <div>
              <strong>{imageDraft.filename || '图片'}</strong>
              <span>{formatBytes(imageDraft.size_bytes)}</span>
            </div>
            <button type="button" className="icon-button" onClick={clearImageDraft} title="移除图片" aria-label="移除图片">
              <X size={15} />
            </button>
          </div>
        )}
        <textarea
          rows="3"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
            placeholder="输入问题"
          disabled={streaming}
        />
        <div className="composer-actions">
          <span>{sendState}</span>
          <div className="composer-buttons">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
              onChange={handleImageChange}
              hidden
            />
            <button
              type="button"
              className="icon-button"
              onClick={() => imageInputRef.current?.click()}
              disabled={streaming}
              title="上传图片提问"
              aria-label="上传图片提问"
            >
              <ImagePlus size={16} />
            </button>
            <button className="primary" type="submit" disabled={streaming || (!input.trim() && !imageDraft)}>
              <Send size={15} />
              发送
            </button>
          </div>
        </div>
      </form>
    </section>
  );
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}
