import React, { useState, useRef, useEffect } from 'react';
import { Building2, Check, Globe2, ImagePlus, Landmark, Moon, PanelLeftOpen, Send, ShieldCheck, Sun, ThumbsDown, ThumbsUp, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SourceEvidenceList from './SourceEvidenceList.jsx';
import ToolCallTimeline from './ToolCallTimeline.jsx';

export default function ChatPane({
  messages, streaming, sendState, onSend, ctx, setCtx, audit, feedbackResult, onFeedback,
  onLoadToolTrace,
  onResolveConfirmation,
  theme = 'light', onToggleTheme,
  sidebarCollapsed = false, onToggleSidebar,
}) {
  const [input, setInput] = useState('');
  const [imageDraft, setImageDraft] = useState(null);
  const [autoFollow, setAutoFollow] = useState(true);
  const messagesRef = useRef(null);
  const messagesEnd = useRef(null);
  const previousCount = useRef(messages.length);
  const imageInputRef = useRef(null);
  const confirmationPending = messages.some((message) =>
    ['pending', 'submitting'].includes(message.confirmation?.status)
  );

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
    if ((!q && !imageDraft) || streaming || confirmationPending) return;
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
    <section className="chat-pane">
      {/* 页头 */}
      <div className="chat-header">
        <div className="chat-heading">
          {sidebarCollapsed && (
            <button className="icon-button" onClick={onToggleSidebar} title="展开侧栏" aria-label="展开侧栏">
              <PanelLeftOpen size={16} />
            </button>
          )}
          <h2>政务智办</h2>
        </div>
        <div className="chat-actions">
          <button
            type="button"
            className="icon-button"
            onClick={onToggleTheme}
            title={theme === 'dark' ? '切换到浅色模式' : '切换到黑暗模式'}
            aria-label={theme === 'dark' ? '切换到浅色模式' : '切换到黑暗模式'}
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </div>
      </div>

      {/* 消息 */}
      <div className="messages" ref={messagesRef} onScroll={handleMessagesScroll}>
        {messages.length === 0 && (
          <div className="chat-empty-state">
            <span><Landmark size={24} /></span>
            <h1>Gov Service Agent</h1>
            <p>今天想办理什么政务事项？</p>
          </div>
        )}
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
              m.role === 'assistant' ? (
                <div className="message-content markdown-body">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={markdownComponents}
                  >
                    {hideInlineCitationMarkers(m.content)}
                  </ReactMarkdown>
                </div>
              ) : (
                <div className="message-content">{m.content}</div>
              )
            )}
            {m.role === 'assistant' && m.sources?.length > 0 && (
              <SourceEvidenceList sources={m.sources} />
            )}
            {m.role === 'assistant' && i === lastAssistantIndex && m.suggestedActions?.some(
              (action) => action.type === 'select_service_channel'
            ) && (
              <ServiceChannelSelection
                actions={m.suggestedActions.filter((action) => action.type === 'select_service_channel')}
                disabled={streaming}
                onSelect={(action) => onSend?.(
                  action.payload?.query || action.label,
                  action.label,
                )}
              />
            )}
            {m.role === 'assistant' && m.confirmation && (
              <HumanConfirmationCard
                confirmation={m.confirmation}
                disabled={streaming}
                onResolve={(approved, selectedSlotIds) => onResolveConfirmation?.(
                  i,
                  m.confirmation,
                  approved,
                  selectedSlotIds,
                )}
              />
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

      {/* 输入区 */}
      <form className="composer" onSubmit={handleSubmit}>
        <div className="composer-box">
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
            rows="2"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={confirmationPending ? '请先处理上方人工确认' : '输入问题'}
            disabled={streaming || confirmationPending}
          />
          <div className={`composer-actions${sendState ? '' : ' is-idle'}`}>
            {sendState && <span className="composer-status"><i />{sendState}</span>}
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
                disabled={streaming || confirmationPending}
                title="上传图片提问"
                aria-label="上传图片提问"
              >
                <ImagePlus size={16} />
              </button>
              <button className="primary send-button" type="submit" disabled={streaming || confirmationPending || (!input.trim() && !imageDraft)} title="发送" aria-label="发送">
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      </form>
    </section>
  );
}

function ServiceChannelSelection({ actions, disabled, onSelect }) {
  return (
    <section className="service-channel-selection" aria-label="选择办理方式">
      <div className="service-channel-selection-head">
        <ShieldCheck size={17} />
        <div>
          <strong>请选择办理方式</strong>
          <span>您的选择将决定后续进入在线申报或大厅预约。</span>
        </div>
      </div>
      <div className="service-channel-options">
        {actions.map((action) => {
          const online = action.payload?.channel === 'online';
          const Icon = online ? Globe2 : Building2;
          return (
            <button
              key={action.payload?.channel || action.label}
              type="button"
              className={`service-channel-option channel-${action.payload?.channel || 'unknown'}`}
              disabled={disabled}
              onClick={() => onSelect(action)}
            >
              <Icon size={19} />
              <span>
                <strong>{action.label}</strong>
                <small>{action.payload?.description || ''}</small>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function HumanConfirmationCard({ confirmation, disabled, onResolve }) {
  const [selectedSlotIds, setSelectedSlotIds] = useState(() => initialSlotSelections(confirmation));
  useEffect(() => {
    setSelectedSlotIds(initialSlotSelections(confirmation));
  }, [confirmation.interrupt_id]);
  const pending = confirmation.status === 'pending';
  const submitting = confirmation.status === 'submitting';
  const resolvedLabel = confirmation.status === 'approved'
    ? '已确认并执行'
    : confirmation.status === 'rejected'
      ? '已拒绝执行'
      : confirmation.status === 'failed'
        ? '处理失败'
        : '';
  return (
    <section className={`human-confirmation status-${confirmation.status || 'pending'}`}>
      <div className="human-confirmation-head">
        <ShieldCheck size={18} />
        <div>
          <strong>{confirmation.title || '请确认操作'}</strong>
          <span>{confirmation.warning || '确认后系统才会执行。'}</span>
        </div>
      </div>
      <div className="human-confirmation-actions-list">
        {(confirmation.actions || []).map((action) => {
          const options = action.slot_options || [];
          const selectedSlotId = Number(selectedSlotIds[action.call_id] || action.arguments?.slot_id || 0);
          const selectedOption = options.find((option) => Number(option.slot_id) === selectedSlotId) || options[0];
          const halls = uniqueSlotHalls(options);
          const hallOptions = options.filter((option) => option.hall_code === selectedOption?.hall_code);
          return (
            <div key={action.call_id || action.subject} className="human-confirmation-action">
              <strong>{action.subject || action.label}</strong>
              {options.length > 0 ? (
                <div className="appointment-selection-grid">
                  <label>
                    <span>预约大厅</span>
                    <select
                      value={selectedOption?.hall_code || ''}
                      disabled={disabled || submitting}
                      onChange={(event) => {
                        const firstSlot = options.find((option) => option.hall_code === event.target.value);
                        if (firstSlot) {
                          setSelectedSlotIds((current) => ({ ...current, [action.call_id]: Number(firstSlot.slot_id) }));
                        }
                      }}
                    >
                      {halls.map((hall) => (
                        <option key={hall.hall_code} value={hall.hall_code}>{hall.hall_name}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span>预约日期和时段</span>
                    <select
                      value={selectedOption?.slot_id || ''}
                      disabled={disabled || submitting}
                      onChange={(event) => setSelectedSlotIds((current) => ({
                        ...current,
                        [action.call_id]: Number(event.target.value),
                      }))}
                    >
                      {hallOptions.map((option) => (
                        <option key={option.slot_id} value={option.slot_id}>
                          {slotOptionLabel(option)}
                        </option>
                      ))}
                    </select>
                  </label>
                  {selectedOption?.hall_address && <span className="appointment-selection-address">{selectedOption.hall_address}</span>}
                </div>
              ) : (
                (action.details || []).map((detail) => <span key={detail}>{detail}</span>)
              )}
            </div>
          );
        })}
      </div>
      {confirmation.error && <div className="human-confirmation-error">{confirmation.error}</div>}
      {(pending || submitting) ? (
        <div className="human-confirmation-buttons">
          <button type="button" className="confirmation-reject" disabled={disabled || submitting} onClick={() => onResolve(false, {})}>
            <X size={14} />取消
          </button>
          <button type="button" className="primary confirmation-approve" disabled={disabled || submitting} onClick={() => onResolve(true, selectedSlotIds)}>
            <Check size={14} />{submitting ? '处理中…' : '确认执行'}
          </button>
        </div>
      ) : (
        <div className="human-confirmation-result">{resolvedLabel}</div>
      )}
    </section>
  );
}

function initialSlotSelections(confirmation) {
  return Object.fromEntries(
    (confirmation.actions || [])
      .filter((action) => action.tool_name === 'appointment_book' && action.arguments?.slot_id)
      .map((action) => [action.call_id, Number(action.arguments.slot_id)]),
  );
}

function uniqueSlotHalls(options) {
  const halls = new Map();
  for (const option of options) {
    if (!halls.has(option.hall_code)) {
      halls.set(option.hall_code, {
        hall_code: option.hall_code,
        hall_name: option.hall_name || option.hall_code,
      });
    }
  }
  return [...halls.values()];
}

function slotOptionLabel(option) {
  const time = [option.start_time, option.end_time].filter(Boolean).join('–');
  return `${option.service_date || ''} ${time}（剩余 ${Number(option.remaining_count || 0)} 个）`.trim();
}

const markdownComponents = {
  a: ({ node: _node, ...props }) => (
    <a {...props} target="_blank" rel="noreferrer" />
  ),
  table: ({ node: _node, ...props }) => (
    <div className="markdown-table-wrap">
      <table {...props} />
    </div>
  ),
};

function findLastAssistantIndex(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'assistant') return index;
  }
  return -1;
}

function hideInlineCitationMarkers(content) {
  return String(content || '').replace(/\[(?:证据|业务)\s*\d+\]/g, '');
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}
