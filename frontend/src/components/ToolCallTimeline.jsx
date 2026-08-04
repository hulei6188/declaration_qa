import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  CircleMinus,
  FileSearch,
  ListChecks,
  LoaderCircle,
  Search,
  XCircle,
} from 'lucide-react';

const TOOL_LABELS = {
  knowledge_search: '查询办事资料',
  ticket_list_query: '查询申请事项',
  ticket_detail_query: '查询申请详情',
  appointment_list_query: '查询预约记录',
  appointment_hall_query: '查询预约大厅',
  appointment_slot_query: '查询预约号源',
  payment_order_query: '查询缴费订单',
  payment_receipt_query: '查询电子票据',
  submission_receipt_query: '查询申请回执',
  result_certificate_query: '查询办结结果',
  delivery_status_query: '查询送达状态',
  queue_status_query: '查询排队状态',
  consultation_record_query: '查询咨询记录',
  application_submit: '提交申报',
  complaint_create: '创建投诉',
  appointment_book: '提交预约',
  appointment_cancel: '取消预约',
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
  const activeTrace = trace || { status: 'idle', loaded: false, tools: [] };
  const tools = activeTrace.tools || [];
  const failed = activeTrace.status === 'failed';
  const running = activeTrace.status === 'running' || activeTrace.status === 'loading';

  function toggle() {
    const next = !expanded;
    setExpanded(next);
    if (next && runId && !activeTrace.loaded && activeTrace.status !== 'loading') onLoad?.();
  }

  return (
    <section className={`tool-trace ${failed ? 'is-failed' : ''}`} aria-label="查询记录">
      <button type="button" className="tool-trace-toggle" onClick={toggle} aria-expanded={expanded}>
        <span className={`tool-trace-state ${running ? 'is-running' : ''}`}>
          {running ? <LoaderCircle size={15} /> : failed ? <XCircle size={15} /> : <Activity size={15} />}
        </span>
        <span className="tool-trace-title">{traceTitle(activeTrace, tools)}</span>
        {tools.length > 0 && <span className="tool-trace-count">{tools.length}</span>}
        <ChevronDown className={expanded ? 'is-expanded' : ''} size={15} />
      </button>

      {expanded && (
        <div className="tool-trace-body">
          {activeTrace.status === 'loading' && <div className="tool-trace-empty">正在读取查询记录</div>}
          {activeTrace.error && <div className="tool-trace-error">{activeTrace.error}</div>}
          {tools.map((tool, index) => (
            <ToolStep key={tool.callId || `${tool.toolName}-${index}`} tool={tool} />
          ))}
          {!running && !activeTrace.error && tools.length === 0 && activeTrace.loaded && (
            <div className="tool-trace-empty">本次问题无需查询外部信息</div>
          )}
        </div>
      )}
    </section>
  );
}

function ToolStep({ tool }) {
  const Icon = toolIcon(tool.toolName);
  const details = argumentDetails(tool.toolName, tool.arguments);
  const label = TOOL_LABELS[tool.toolName] || '处理业务信息';
  return (
    <div className={`tool-step status-${tool.status || 'running'} ${tool.parentWorkerId ? 'is-worker-child' : ''}`}>
      <span className="tool-step-icon"><Icon size={14} /></span>
      <div className="tool-step-main">
        <div className="tool-step-head">
          <strong>{label}</strong>
          <span>{statusIcon(tool.status)}{statusLabel(tool.status)}</span>
        </div>
        {details.length > 0 && (
          <div className="tool-step-details">
            {details.map((detail, index) => <span key={`${detail}-${index}`}>{detail}</span>)}
          </div>
        )}
        <div className="tool-step-meta">
          {tool.iteration > 0 && <span>第 {tool.iteration} 步</span>}
          {tool.parallel && <span>同时处理</span>}
          {tool.latencyMs !== null && tool.latencyMs !== undefined && <span>{formatLatency(tool.latencyMs)}</span>}
          {tool.sourceCount > 0 && <span>{tool.sourceCount} 条参考信息</span>}
          {tool.failureReason && <span>{failureReasonLabel(tool.failureReason)}</span>}
        </div>
      </div>
    </div>
  );
}

function argumentDetails(toolName, args = {}) {
  if (toolName === 'knowledge_search') {
    return (args.queries || []).map((query) => {
      const text = typeof query === 'string' ? query : query?.text;
      return text;
    }).filter(Boolean);
  }
  const details = [];
  if (args.item_code) details.push('已匹配办理事项');
  if (Array.isArray(args.item_codes)) details.push(`已匹配 ${args.item_codes.length} 个办理事项`);
  if (Array.isArray(args.hall_codes)) details.push(`正在查询 ${args.hall_codes.length} 个预约大厅`);
  if (args.application_id !== undefined && args.application_id !== null) details.push(`申请编号：${args.application_id}`);
  if (Array.isArray(args.application_ids)) details.push(`申请编号：${args.application_ids.join('、')}`);
  if (Array.isArray(args.targets)) {
    if (args.targets.length > 0) details.push(`正在查询 ${args.targets.length} 笔申请或缴费信息`);
  }
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
  if (status === 'empty') return <CircleMinus size={13} />;
  if (status === 'awaiting_confirmation') return <Activity size={13} />;
  if (status === 'rejected') return <XCircle size={13} />;
  if (status === 'failed') return <XCircle size={13} />;
  return <LoaderCircle className="is-spinning" size={13} />;
}

function statusLabel(status) {
  if (status === 'completed') return '已完成';
  if (status === 'empty') return '无记录';
  if (status === 'awaiting_confirmation') return '等待确认';
  if (status === 'rejected') return '已拒绝';
  if (status === 'failed') return '失败';
  return '执行中';
}

function traceTitle(trace, tools = []) {
  if (trace.status === 'loading') return '正在读取查询记录';
  if (trace.status === 'failed') return '查询未完成';
  if (trace.status === 'paused' || trace.phase === 'awaiting_confirmation') return '等待您确认';
  if (trace.status === 'running') {
    if (trace.phase === 'answering') return '正在整理答复';
    if (trace.phase === 'checking') return '正在核对查询结果';
    if (tools.length > 0) return runningTraceTitle(tools);
    return '正在理解您的问题';
  }
  return completedTraceTitle(tools);
}

function runningTraceTitle(tools) {
  const categories = toolCategories(tools);
  if (categories.size === 1 && categories.has('knowledge')) return '正在查询办事资料';
  if (categories.has('write')) return '正在处理业务操作';
  if (categories.size === 1 && categories.has('business')) return '正在查询业务信息';
  return '正在查询相关信息';
}

function completedTraceTitle(tools) {
  const categories = toolCategories(tools);
  if (categories.size === 1 && categories.has('knowledge')) return '查看资料检索';
  if (categories.has('write')) return '查看操作记录';
  if (categories.size === 1 && categories.has('business')) return '查看业务查询';
  return '查看查询记录';
}

function toolCategories(tools) {
  const categories = new Set();
  for (const tool of tools) {
    const name = String(tool?.toolName || '');
    if (name === 'knowledge_search') categories.add('knowledge');
    else if (['appointment_book', 'appointment_cancel', 'application_submit', 'complaint_create'].includes(name)) categories.add('write');
    else categories.add('business');
  }
  return categories;
}

function formatLatency(value) {
  const ms = Number(value || 0);
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} 秒`;
  return `${Math.round(ms)} 毫秒`;
}

function failureReasonLabel(reason) {
  if (reason === 'worker_failed_to_plan_query') return '未找到合适的查询方式';
  if (reason === 'evidence_coverage_incomplete') return '参考信息暂不完整';
  return '处理未完成';
}
