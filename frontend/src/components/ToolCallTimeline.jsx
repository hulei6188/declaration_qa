import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  FileSearch,
  ListChecks,
  LoaderCircle,
  Search,
  ShieldCheck,
  XCircle,
} from 'lucide-react';

const TOOL_LABELS = {
  knowledge_search: '检索政策知识',
  ticket_list_query: '查询申请事项',
  ticket_detail_query: '查询申请详情',
};

const EVIDENCE_LABELS = {
  knowledge: '知识证据',
  ticket_list: '申请列表',
  ticket_detail: '申请详情',
};

export default function ToolCallTimeline({ trace, runId, onLoad }) {
  const isLive = trace?.status === 'running';
  const [expanded, setExpanded] = useState(isLive);
  const wasLive = useRef(isLive);

  useEffect(() => {
    if (isLive) setExpanded(true);
    else if (wasLive.current && trace?.status === 'completed') setExpanded(false);
    wasLive.current = isLive;
  }, [isLive, trace?.status]);

  if (!trace && !runId) return null;
  const activeTrace = trace || { status: 'idle', loaded: false, tools: [], gates: [] };
  const tools = activeTrace.tools || [];
  const failed = activeTrace.status === 'failed';
  const running = activeTrace.status === 'running' || activeTrace.status === 'loading';

  function toggle() {
    const next = !expanded;
    setExpanded(next);
    if (next && runId && !activeTrace.loaded && activeTrace.status !== 'loading') onLoad?.();
  }

  return (
    <section className={`tool-trace ${failed ? 'is-failed' : ''}`} aria-label="Agent 工具执行过程">
      <button type="button" className="tool-trace-toggle" onClick={toggle} aria-expanded={expanded}>
        <span className={`tool-trace-state ${running ? 'is-running' : ''}`}>
          {running ? <LoaderCircle size={15} /> : failed ? <XCircle size={15} /> : <Activity size={15} />}
        </span>
        <span className="tool-trace-title">{traceTitle(activeTrace, tools.length)}</span>
        {tools.length > 0 && <span className="tool-trace-count">{tools.length}</span>}
        <ChevronDown className={expanded ? 'is-expanded' : ''} size={15} />
      </button>

      {expanded && (
        <div className="tool-trace-body">
          {activeTrace.status === 'loading' && <div className="tool-trace-empty">正在读取执行记录</div>}
          {activeTrace.error && <div className="tool-trace-error">{activeTrace.error}</div>}
          {tools.map((tool, index) => (
            <ToolStep key={tool.callId || `${tool.toolName}-${index}`} tool={tool} />
          ))}
          {(activeTrace.gates || []).map((gate, index) => (
            <EvidenceStep key={`${gate.iteration}-${index}`} gate={gate} />
          ))}
          {!running && !activeTrace.error && tools.length === 0 && activeTrace.loaded && (
            <div className="tool-trace-empty">本次回答未调用外部工具</div>
          )}
        </div>
      )}
    </section>
  );
}

function ToolStep({ tool }) {
  const Icon = toolIcon(tool.toolName);
  const details = argumentDetails(tool.toolName, tool.arguments);
  return (
    <div className={`tool-step status-${tool.status || 'running'}`}>
      <span className="tool-step-icon"><Icon size={14} /></span>
      <div className="tool-step-main">
        <div className="tool-step-head">
          <strong>{TOOL_LABELS[tool.toolName] || tool.toolName || '执行工具'}</strong>
          <span>{statusIcon(tool.status)}{statusLabel(tool.status)}</span>
        </div>
        {details.length > 0 && (
          <div className="tool-step-details">
            {details.map((detail, index) => <span key={`${detail}-${index}`}>{detail}</span>)}
          </div>
        )}
        <div className="tool-step-meta">
          {tool.iteration > 0 && <span>第 {tool.iteration} 轮</span>}
          {tool.parallel && <span>并行执行</span>}
          {tool.dependency && <span>依赖上一项结果</span>}
          {tool.latencyMs !== null && tool.latencyMs !== undefined && <span>{formatLatency(tool.latencyMs)}</span>}
        </div>
      </div>
    </div>
  );
}

function EvidenceStep({ gate }) {
  const checks = gate.required?.length ? gate.required : Object.keys(gate.checks || {});
  return (
    <div className={`tool-step evidence-step ${gate.ok ? 'status-completed' : 'status-failed'}`}>
      <span className="tool-step-icon"><ShieldCheck size={14} /></span>
      <div className="tool-step-main">
        <div className="tool-step-head">
          <strong>证据完整性检查</strong>
          <span>{gate.ok ? <CheckCircle2 size={13} /> : <XCircle size={13} />}{gate.ok ? '通过' : '待补充'}</span>
        </div>
        <div className="evidence-checks">
          {checks.map((key) => (
            <span key={key} className={gate.checks?.[key] ? 'is-ok' : 'is-missing'}>
              {gate.checks?.[key] ? <CheckCircle2 size={11} /> : <CircleDashed size={11} />}
              {EVIDENCE_LABELS[key] || key}
            </span>
          ))}
        </div>
        <div className="tool-step-meta">
          {gate.iteration > 0 && <span>第 {gate.iteration} 轮</span>}
          <span>{gate.sourceCount} 条知识证据</span>
          <span>{gate.toolResultCount} 个工具结果</span>
        </div>
      </div>
    </div>
  );
}

function argumentDetails(toolName, args = {}) {
  if (toolName === 'knowledge_search') {
    return (args.queries || []).map((query) => {
      const text = typeof query === 'string' ? query : query?.text;
      const strategy = typeof query === 'object' ? query?.strategy : '';
      return [text, strategy && `(${strategy})`].filter(Boolean).join(' ');
    }).filter(Boolean);
  }
  const details = [];
  if (args.item_code) details.push(`事项编码：${args.item_code}`);
  if (args.application_id !== undefined && args.application_id !== null) details.push(`申请 ID：${args.application_id}`);
  if (args.query) details.push(String(args.query));
  return details;
}

function toolIcon(name) {
  if (name === 'knowledge_search') return Search;
  if (name === 'ticket_list_query') return ListChecks;
  if (name === 'ticket_detail_query') return FileSearch;
  return Activity;
}

function statusIcon(status) {
  if (status === 'completed') return <CheckCircle2 size={13} />;
  if (status === 'failed') return <XCircle size={13} />;
  return <LoaderCircle className="is-spinning" size={13} />;
}

function statusLabel(status) {
  if (status === 'completed') return '已完成';
  if (status === 'failed') return '失败';
  return '执行中';
}

function traceTitle(trace, count) {
  if (trace.status === 'loading') return '正在读取执行过程';
  if (trace.status === 'failed') return '工具执行未完成';
  if (trace.status === 'running') {
    if (trace.phase === 'answering') return '证据已就绪，正在生成回答';
    if (trace.phase === 'checking') return '正在检查工具结果';
    if (count > 0) return '正在调用工具';
    return '正在规划工具';
  }
  return '查看工具调用过程';
}

function formatLatency(value) {
  const ms = Number(value || 0);
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} 秒`;
  return `${Math.round(ms)} 毫秒`;
}
