import React, { useState, useRef, useEffect } from 'react';
import { ArrowRight, Building2, Check, ClipboardList, FileText, FileUp, Globe2, ImagePlus, Landmark, Moon, PanelLeftOpen, Send, ShieldCheck, Sun, ThumbsDown, ThumbsUp, Trash2, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SourceEvidenceList from './SourceEvidenceList.jsx';
import ToolCallTimeline from './ToolCallTimeline.jsx';

export default function ChatPane({
  messages, streaming, sendState, onSend, ctx, setCtx, audit, feedbackResult, onFeedback,
  onLoadToolTrace,
  onResolveConfirmation,
  onUploadMaterial,
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
    if ((!q && !imageDraft) || streaming || confirmationPending || formFillingPending) return;
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
  const activeDialogTask = messages[lastAssistantIndex]?.dialogTask || {};
  const formFillingPending = activeDialogTask.status === 'collecting' && Boolean(activeDialogTask.next_field?.name);
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
            {m.role === 'assistant' && i === lastAssistantIndex && m.dialogTask?.status === 'collecting' && m.dialogTask?.next_field?.name && (
              <SlotFillingCard
                task={m.dialogTask}
                disabled={streaming}
                onUploadMaterial={onUploadMaterial}
                onCancel={() => onSend?.('取消当前表单填写流程', '取消填写')}
                onSubmit={(field, value) => onSend?.(
                  buildSlotSubmissionQuery(m.dialogTask, field, value),
                  slotSubmissionDisplay(field, value),
                  {
                    formSubmission: {
                      task_type: m.dialogTask.task_type,
                      field_name: field.name,
                      value,
                    },
                  },
                )}
              />
            )}
            {m.role === 'assistant' && m.confirmation && confirmationCardIsVisible(m.confirmation) && (
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
            placeholder={confirmationPending
              ? '请先处理上方人工确认'
              : formFillingPending
                ? '请在上方完成表单填写，如需退出请点击“取消填写”'
                : '输入问题'}
            disabled={streaming || confirmationPending || formFillingPending}
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
                disabled={streaming || confirmationPending || formFillingPending}
                title="上传图片提问"
                aria-label="上传图片提问"
              >
                <ImagePlus size={16} />
              </button>
              <button className="primary send-button" type="submit" disabled={streaming || confirmationPending || formFillingPending || (!input.trim() && !imageDraft)} title="发送" aria-label="发送">
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

function SlotFillingCard({ task, disabled, onSubmit, onCancel, onUploadMaterial }) {
  const field = task.next_field || {};
  const initialValue = initialFieldValue(field, task.values || {});
  const [value, setValue] = useState(initialValue);
  const [error, setError] = useState('');
  const taskKey = [task.schema_id, task.schema_version, field.name, field.error].join(':');

  useEffect(() => {
    setValue(initialFieldValue(field, task.values || {}));
    setError(field.error || '');
  }, [taskKey]);

  const fields = task.schema?.fields || [];
  const requiredFields = fields.filter((item) => item.required);
  const progressFields = requiredFields.length > 0 ? requiredFields : fields;
  const completed = progressFields.filter((item) => hasSlotValue(task.values?.[item.name])).length;
  const total = Math.max(progressFields.length, 1);
  const progress = Math.min(100, Math.round((completed / total) * 100));
  const formTitle = task.task_type === 'complaint_create'
    ? '投诉信息填写'
    : `${task.item_name || '事项'}在线申报`;

  function handleSubmit(event) {
    event.preventDefault();
    const validationError = validateSlotField(field, value);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError('');
    onSubmit(field, normalizedSlotValue(field, value));
  }

  return (
    <section className="slot-filling-card" aria-label="逐项填写表单">
      <div className="slot-filling-head">
        <ClipboardList size={18} />
        <div>
          <strong>{formTitle}</strong>
          <span>系统会保存已填内容，并逐项引导完成剩余字段。</span>
        </div>
        <b>{completed}/{total}</b>
      </div>
      <div
        className="slot-progress"
        role="progressbar"
        aria-label="表单填写进度"
        aria-valuemin="0"
        aria-valuemax={total}
        aria-valuenow={completed}
      >
        <i style={{ width: `${progress}%` }} />
      </div>
      <form className="slot-field-form" onSubmit={handleSubmit}>
        <label className="slot-field-label" htmlFor={`slot-field-${field.name}`}>
          <span>{field.label || field.name}{field.required && <em>必填</em>}</span>
          {field.description && <small>{field.description}</small>}
        </label>
        <SlotFieldControl
          task={task}
          field={field}
          value={value}
          disabled={disabled}
          onUploadMaterial={onUploadMaterial}
          onChange={(nextValue) => {
            setValue(nextValue);
            if (error) setError('');
          }}
        />
        {error && <div className="slot-field-error">{error}</div>}
        <div className="slot-field-actions">
          <span>下一步将继续填写下一个必填项</span>
          <div className="slot-field-action-buttons">
            <button type="button" className="ghost" disabled={disabled} onClick={onCancel}>
              <X size={14} />取消填写
            </button>
            <button type="submit" className="primary" disabled={disabled || !hasSlotValue(value)}>
              {disabled ? '提交中…' : '保存并继续'}<ArrowRight size={14} />
            </button>
          </div>
        </div>
      </form>
    </section>
  );
}

function SlotFieldControl({ task, field, value, disabled, onChange, onUploadMaterial }) {
  const type = String(field.type || 'text').toLowerCase();
  const options = field.options || [];
  const inputId = `slot-field-${field.name}`;

  if (type === 'material_group') {
    return (
      <MaterialUploadControl
        task={task}
        field={field}
        value={value}
        disabled={disabled}
        onChange={onChange}
        onUpload={onUploadMaterial}
      />
    );
  }

  if (type === 'textarea') {
    return (
      <textarea
        id={inputId}
        className="slot-input slot-textarea"
        rows="4"
        value={value}
        minLength={field.min_length || undefined}
        maxLength={field.max_length || undefined}
        disabled={disabled}
        autoFocus
        onChange={(event) => onChange(event.target.value)}
      />
    );
  }

  if (['select', 'radio', 'multiselect', 'checkbox', 'boolean'].includes(type)) {
    const choices = type === 'boolean' ? ['是', '否'] : options;
    const multiple = ['multiselect', 'checkbox'].includes(type);
    const selected = multiple ? (Array.isArray(value) ? value : []) : [String(value || '')];
    return (
      <div id={inputId} className="slot-choice-grid" role={multiple ? 'group' : 'radiogroup'}>
        {choices.map((option) => {
          const checked = selected.includes(String(option));
          return (
            <button
              key={option}
              type="button"
              className={checked ? 'is-selected' : ''}
              aria-pressed={checked}
              disabled={disabled}
              onClick={() => {
                if (!multiple) {
                  onChange(String(option));
                  return;
                }
                onChange(checked
                  ? selected.filter((item) => item !== String(option))
                  : [...selected, String(option)]);
              }}
            >
              <i>{checked && <Check size={13} />}</i>
              {option}
            </button>
          );
        })}
      </div>
    );
  }

  const htmlType = type === 'date'
    ? 'date'
    : ['number', 'integer'].includes(type)
      ? 'number'
      : type === 'phone'
        ? 'tel'
        : 'text';
  return (
    <input
      id={inputId}
      className="slot-input"
      type={htmlType}
      inputMode={type === 'phone' ? 'tel' : ['number', 'integer'].includes(type) ? 'decimal' : undefined}
      step={type === 'number' ? 'any' : type === 'integer' ? '1' : undefined}
      value={value}
      minLength={field.min_length || undefined}
      maxLength={field.max_length || undefined}
      disabled={disabled}
      autoFocus
      autoComplete={type === 'phone' ? 'tel' : 'off'}
      placeholder={slotFieldPlaceholder(field)}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

function MaterialUploadControl({ task, field, value, disabled, onChange, onUpload }) {
  const [uploadingKey, setUploadingKey] = useState('');
  const [uploadError, setUploadError] = useState('');
  const group = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const accept = (field.accept || []).join(',');

  async function uploadFiles(material, files) {
    if (!files.length || disabled || uploadingKey) return;
    const current = group[material.key] || [];
    const remaining = Math.max(0, Number(material.max_files || 1) - current.length);
    if (remaining === 0) {
      setUploadError(`“${material.label}”最多上传 ${material.max_files || 1} 个文件`);
      return;
    }
    setUploadingKey(material.key);
    setUploadError('');
    try {
      const uploaded = [];
      for (const file of files.slice(0, remaining)) {
        uploaded.push(await onUpload(file, task, field, material));
      }
      onChange({ ...group, [material.key]: [...current, ...uploaded] });
    } catch (error) {
      setUploadError(error.message || '材料上传失败');
    } finally {
      setUploadingKey('');
    }
  }

  return (
    <div className="material-upload-list">
      {(field.materials || []).map((material) => {
        const files = group[material.key] || [];
        const isUploading = uploadingKey === material.key;
        return (
          <section key={material.key} className="material-upload-item">
            <div className="material-upload-title">
              <div>
                <strong>{material.label}</strong>
                {material.required ? <em>必传</em> : <span>按情形上传</span>}
              </div>
              {material.description && <small>{material.description}</small>}
            </div>
            {files.length > 0 && (
              <div className="material-files">
                {files.map((file) => (
                  <div key={file.attachment_id} className="material-file">
                    <FileText size={15} />
                    <span><strong>{file.filename}</strong><small>{formatBytes(file.size_bytes)}</small></span>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label={`移除${file.filename}`}
                      title="移除文件"
                      disabled={disabled || Boolean(uploadingKey)}
                      onClick={() => onChange({
                        ...group,
                        [material.key]: files.filter((item) => item.attachment_id !== file.attachment_id),
                      })}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <label className={`material-upload-button${isUploading ? ' is-uploading' : ''}`}>
              <FileUp size={15} />
              {isUploading ? '正在上传…' : files.length ? '继续添加' : '选择文件'}
              <input
                type="file"
                accept={accept}
                multiple={Number(material.max_files || 1) > 1}
                disabled={disabled || Boolean(uploadingKey)}
                onChange={(event) => {
                  const selected = Array.from(event.target.files || []);
                  event.target.value = '';
                  uploadFiles(material, selected);
                }}
              />
            </label>
          </section>
        );
      })}
      {uploadError && <div className="slot-field-error">{uploadError}</div>}
      <p className="material-upload-hint">支持 PDF、Word、Excel、JPG、PNG、ZIP，单个文件不超过 20 MB。</p>
    </div>
  );
}

function initialFieldValue(field, values) {
  const existing = values?.[field.name];
  if (existing !== undefined && existing !== null) return existing;
  if (String(field.type || '').toLowerCase() === 'material_group') return {};
  return ['multiselect', 'checkbox'].includes(String(field.type || '').toLowerCase()) ? [] : '';
}

function hasSlotValue(value) {
  if (Array.isArray(value)) return value.length > 0;
  if (value && typeof value === 'object') return Object.values(value).some((item) => hasSlotValue(item));
  return value !== undefined && value !== null && String(value).trim() !== '';
}

function normalizedSlotValue(field, value) {
  const type = String(field.type || '').toLowerCase();
  if (type === 'material_group') return value && typeof value === 'object' ? value : {};
  if (['multiselect', 'checkbox'].includes(type)) return Array.isArray(value) ? value : [];
  if (type === 'boolean') return String(value) === '是';
  return String(value).trim();
}

function validateSlotField(field, value) {
  if (field.required && !hasSlotValue(value)) return `请填写${field.label || '该字段'}`;
  if (field.type === 'material_group') {
    for (const material of field.materials || []) {
      if (material.required && !(value?.[material.key] || []).length) return `请上传“${material.label}”`;
    }
    return '';
  }
  const text = Array.isArray(value) ? value.join(',') : String(value ?? '').trim();
  if (field.min_length && text.length < Number(field.min_length)) return `至少填写 ${field.min_length} 个字符`;
  if (field.max_length && text.length > Number(field.max_length)) return `最多填写 ${field.max_length} 个字符`;
  if (field.type === 'phone' && !/^(?:(?:\+?86[- ]?)?1\d{10}|0\d{2,3}[- ]?\d{7,8})$/.test(text)) return '请输入有效联系电话';
  if (field.type === 'id_number' && !/^(?:\d{17}[\dXx]|[A-Za-z0-9]{6,30})$/.test(text)) return '请输入有效证件号码';
  return '';
}

function slotFieldPlaceholder(field) {
  if (field.type === 'phone') return '请输入手机号或固定电话';
  if (field.type === 'id_number') return '请输入证件号码';
  if (['number', 'integer'].includes(field.type)) return '请输入数字';
  return `请输入${field.label || '内容'}`;
}

function buildSlotSubmissionQuery(task, field, value) {
  const taskLabel = task.task_type === 'complaint_create' ? '投诉' : '申报';
  if (field.type === 'material_group') return `继续填写${taskLabel}表单，已完成申报材料上传`;
  const serialized = Array.isArray(value) ? value.join('、') : value === true ? '是' : value === false ? '否' : String(value);
  return `继续填写${taskLabel}表单，${field.label || field.name}：${serialized}`;
}

function slotSubmissionDisplay(field, value) {
  if (field.type === 'material_group') {
    const count = Object.values(value || {}).reduce((total, files) => total + (Array.isArray(files) ? files.length : 0), 0);
    return `已上传申报材料（${count} 份）`;
  }
  const serialized = Array.isArray(value) ? value.join('、') : value === true ? '是' : value === false ? '否' : String(value);
  const type = String(field.type || '').toLowerCase();
  let displayValue = serialized;
  if (type === 'phone' && serialized.length >= 7) displayValue = `${serialized.slice(0, 3)}****${serialized.slice(-4)}`;
  if (type === 'id_number' && serialized.length >= 8) displayValue = `${serialized.slice(0, 4)}******${serialized.slice(-4)}`;
  return `已填写：${field.label || field.name}（${displayValue}）`;
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

function confirmationCardIsVisible(confirmation) {
  const status = String(confirmation?.status || 'pending');
  return ['pending', 'submitting', 'failed'].includes(status);
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
