import React, { useState } from 'react';
import { Copy, Play, ThumbsUp, ThumbsDown, Package } from 'lucide-react';
import { escapeHtml } from '../utils.js';

export default function Inspector({
  timeline, sources, audit, feedbackResult, evalDatasetId, evalRecall, evalRefusal,
  evalItems, evalResult, onFeedback, onCreateDataset, onRunEval, onAnnotate, onCopyTimeline,
}) {
  const [evalName, setEvalName] = useState('申报问答回归集');

  return (
    <aside className="inspect-pane">
      {/* Timeline */}
      <section className="panel">
        <div className="panel-title">
          <span>Workflow 事件</span>
          <button className="icon-button" onClick={onCopyTimeline} title="复制 Workflow 事件">
            <Copy size={14} />
          </button>
        </div>
        <div className="timeline">
          {timeline.length === 0 ? (
            <p className="hint">暂无事件</p>
          ) : (
            timeline.map((item, i) => (
              <div key={i} className="timeline-item">
                <strong>{item.event}</strong>
                <code>{item.data}</code>
              </div>
            ))
          )}
        </div>
      </section>

      {/* Sources */}
      <section className="panel">
        <div className="panel-title">
          <span>证据来源</span>
        </div>
        <div className="sources">
          {sources.length === 0 ? (
            <p className="hint">暂无来源</p>
          ) : (
            sources.map((s, i) => (
              <div key={i} className="source-card">
                <strong>{escapeHtml(s.title || '未命名来源')}</strong>
                <code>{escapeHtml(s.snippet || '')}</code>
                <code>score={s.score} · {s.region_code} · {s.policy_version || '-'}</code>
              </div>
            ))
          )}
        </div>
      </section>

      {/* Audit */}
      <section className="panel">
        <div className="panel-title">
          <span>审计与反馈</span>
        </div>
        <dl className="audit-grid">
          <dt>意图</dt>
          <dd>{audit.intent}</dd>
          <dt>审计 ID</dt>
          <dd>{audit.auditId}</dd>
          <dt>拒答</dt>
          <dd>{audit.refused}</dd>
        </dl>
        <div className="feedback-row">
          <button onClick={() => onFeedback('up')}>
            <ThumbsUp size={14} />
            有帮助
          </button>
          <button onClick={() => onFeedback('down')}>
            <ThumbsDown size={14} />
            需改进
          </button>
        </div>
        <p className="hint">{feedbackResult}</p>
      </section>

      {/* Eval */}
      <section className="panel">
        <div className="panel-title">
          <span>RAG 质量评估</span>
          <button className="icon-button" onClick={onRunEval} title="运行评估">
            <Play size={14} />
          </button>
        </div>
        <label>
          评测集名称
          <input value={evalName} onChange={(e) => setEvalName(e.target.value)} />
        </label>
        <button className="primary full" onClick={() => onCreateDataset(evalName)}>
          <Package size={14} />
          创建评测集
        </button>
        <dl className="audit-grid">
          <dt>数据集</dt>
          <dd>{evalDatasetId || '-'}</dd>
          <dt>召回率</dt>
          <dd>{evalRecall}</dd>
          <dt>拒答</dt>
          <dd>{evalRefusal}</dd>
        </dl>
        <div className="eval-items">
          {evalItems.map((item) => (
            <div key={item.item_id} className="eval-item">
              <strong>{escapeHtml(item.query)}</strong>
              <code>intent {escapeHtml(item.actual_intent)} · recall {String(item.recall_hit)} · refused {String(item.actual_refused)}</code>
              <div className="inline-actions">
                <button onClick={() => onAnnotate(item.item_id, 'pass')}>通过</button>
                <button onClick={() => onAnnotate(item.item_id, 'fail')}>失败</button>
              </div>
            </div>
          ))}
        </div>
        <p className="hint">{evalResult}</p>
      </section>
    </aside>
  );
}
