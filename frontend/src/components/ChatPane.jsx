import React, { useState, useRef, useEffect } from 'react';
import { ImagePlus, Maximize2, Minimize2, Send, ThumbsDown, ThumbsUp, Trash2, X } from 'lucide-react';
import SourceEvidenceList from './SourceEvidenceList.jsx';
import ToolCallTimeline from './ToolCallTimeline.jsx';

export default function ChatPane({
  messages, streaming, sendState, onSend, onClear, ctx, setCtx, audit, feedbackResult, onFeedback,
  expanded = false, onToggleExpanded, onLoadToolTrace,
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
    onSend(q, q, undefined, undefined, imageDraft ? { ...imageDraft, text_present: Boolean(q) } : null);
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

  const lastAssistantIndex = findLastAssistantIndex(messages);
  const canFeedback = Boolean(audit?.auditId && audit.auditId !== '-');

  return (
    <section className={`chat-pane${expanded ? ' is-expanded' : ''}`}>
      {/* Header */}
      <div className="chat-header">
        <div>
          <p className="eyebrow">Live SSE Chat</p>
          <h2>问答会话</h2>
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
            {m.role === 'assistant' && (m.toolTrace || m.runId) && (
              <ToolCallTimeline
                key={m.runId || `live-trace-${i}`}
                trace={m.toolTrace}
                runId={m.runId}
                onLoad={() => onLoadToolTrace?.(i, m.runId)}
              />
            )}
            {String(m.content || '').trim() && (
              <div className="message-content">{m.content}</div>
            )}
            {m.role === 'assistant' && m.sources?.length > 0 && (
              <SourceEvidenceList sources={m.sources} />
            )}
            {m.attachments?.length > 0 && (
              <div className="message-attachments">
                {m.attachments.map((item, index) => (
                  <img
                    key={`${item.filename || 'image'}-${index}`}
                    src={item.data_url || item.url}
                    alt={item.filename || '上传图片'}
                  />
                ))}
              </div>
            )}
            {m.role === 'assistant' && i === lastAssistantIndex && (
              <div className="message-feedback" aria-label="回答反馈">
                <button type="button" onClick={() => onFeedback?.('up')} disabled={!canFeedback}>
                  <ThumbsUp size={14} />
                  有帮助
                </button>
                <button type="button" onClick={() => onFeedback?.('down')} disabled={!canFeedback}>
                  <ThumbsDown size={14} />
                  需改进
                </button>
                {feedbackResult && <span>{feedbackResult}</span>}
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

function findLastAssistantIndex(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'assistant') return index;
  }
  return -1;
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}
