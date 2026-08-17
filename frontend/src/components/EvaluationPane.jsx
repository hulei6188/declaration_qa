import React, { useEffect, useState } from 'react';
import { CheckCircle2, ClipboardCheck, PanelLeftOpen, Play, RefreshCw, Search, ShieldAlert, X, XCircle } from 'lucide-react';
import {
  annotateEvalItem,
  fetchEvaluationCandidates,
  fetchEvaluationCases,
  fetchEvaluationDatasets,
  fetchEvaluationRun,
  fetchEvaluationRuns,
  formatMetric,
  mineEvaluationCandidates,
  promoteEvaluationCandidate,
  publishEvaluationDataset,
  reviewEvaluationCandidate,
  runEvaluation,
  runEvaluationCase,
  sleep,
} from '../utils.js';

// Keep the API values stable while exposing reviewer-friendly labels in the UI.
const ATTRIBUTION_OPTIONS = [
  { value: 'retrieval_miss', label: '检索未命中', description: '知识库中有内容，但召回失败或排序靠后' },
  { value: 'knowledge_missing', label: '知识缺失', description: '核实后确认知识库没有覆盖该问题' },
  { value: 'intent_mismatch', label: '意图识别错误', description: '路由到错误意图或业务流程' },
  { value: 'slot_error', label: '槽位填充错误', description: '字段提取、校验或多轮状态衔接错误' },
  { value: 'hallucination', label: '事实幻觉', description: '回答编造了政策或业务动态事实' },
  { value: 'tool_error', label: '工具调用异常', description: '工具报错、参数错误或返回异常' },
  { value: 'agent_loop', label: 'Agent 循环失控', description: '重复调用、超迭代或未能收敛' },
  { value: 'stale_policy', label: '政策版本过期', description: '引用了失效或非当前版本的政策' },
  { value: 'unknown', label: '待人工确认', description: '已有失败信号，但暂时无法确定根因' },
];

const ATTRIBUTION_LABELS = Object.fromEntries(ATTRIBUTION_OPTIONS.map((item) => [item.value, item.label]));

function attributionLabel(value) {
  return ATTRIBUTION_LABELS[value] || ATTRIBUTION_LABELS.unknown;
}

function splitFormList(value) {
  return String(value || '').split(/[\n,，、]/).map((item) => item.trim()).filter(Boolean);
}

function jsonFormValue(value, fallback = {}) {
  return JSON.stringify(value || fallback, null, 2);
}

function promotionFormFromCandidate(candidate) {
  return {
    query: candidate.sanitized_query || '',
    expected_intent: '',
    expected_refused: '',
    expected_doc_ids: '',
    expected_chunk_ids: '',
    expected_answer_keywords: '',
    reference_answer: '',
    reference_contexts: '',
    case_type: 'agent',
    expected_trajectory: jsonFormValue({ required_tools: [], forbidden_tools: [], max_tool_calls: 2 }),
    expected_outcome: '{}',
    tags: (candidate.labels || []).filter((tag) => tag !== 'production_failure').join(', '),
    severity: candidate.severity || 'medium',
  };
}

export default function EvaluationPane({ sidebarCollapsed, onToggleSidebar }) {
  const [tab, setTab] = useState('runs');
  const [workspaceView, setWorkspaceView] = useState('cases');
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState('');
  const [cases, setCases] = useState([]);
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [busy, setBusy] = useState(false);
  const [runningCaseId, setRunningCaseId] = useState('');
  const [annotatingItemId, setAnnotatingItemId] = useState('');
  const [promotionCandidate, setPromotionCandidate] = useState(null);
  const [highlightedCaseId, setHighlightedCaseId] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => { loadDatasets(); }, []);
  useEffect(() => {
    if (!datasetId) return;
    setCases([]);
    setRuns([]);
    setSelectedRun(null);
    setHighlightedCaseId('');
    loadCases(datasetId);
    loadRuns(datasetId);
  }, [datasetId]);
  useEffect(() => {
    if (workspaceView === 'runs' && !selectedRun && runs[0]?.run_id) openRun(runs[0].run_id);
  }, [workspaceView, runs, selectedRun]);
  useEffect(() => { if (tab === 'candidates') loadCandidates(); }, [tab]);

  async function loadDatasets() {
    try {
      const values = await fetchEvaluationDatasets();
      setDatasets(values || []);
      setDatasetId((current) => current || values?.[0]?.dataset_id || '');
    } catch (error) { setMessage(`数据集读取失败：${error.message}`); }
  }

  async function loadCases(id = datasetId) {
    if (!id) return;
    try { setCases(await fetchEvaluationCases(id) || []); }
    catch (error) { setMessage(`数据集样本读取失败：${error.message}`); }
  }

  async function loadRuns(id = datasetId) {
    if (!id) return;
    try {
      const values = await fetchEvaluationRuns(id);
      setRuns(values || []);
    } catch (error) { setMessage(`运行记录读取失败：${error.message}`); }
  }

  async function openRun(runId) {
    try { setSelectedRun(await fetchEvaluationRun(runId)); }
    catch (error) { setMessage(`运行详情读取失败：${error.message}`); }
  }

  async function openRunResult(runId) {
    setWorkspaceView('runs');
    await openRun(runId);
  }

  async function annotateItem(itemId, label) {
    if (!itemId || annotatingItemId) return;
    setAnnotatingItemId(itemId);
    try {
      await annotateEvalItem(itemId, label);
      setMessage(label === 'pass' ? '人工标注已保存：通过' : '人工标注已保存：失败');
      if (selectedRun?.run_id) await openRun(selectedRun.run_id);
    } catch (error) {
      setMessage(`人工标注失败：${error.message}`);
    } finally {
      setAnnotatingItemId('');
    }
  }

  async function startRun() {
    if (!datasetId || busy) return;
    setWorkspaceView('runs');
    setBusy(true);
    setRunningCaseId('');
    setMessage('完整数据集评测已提交');
    try {
      const response = await runEvaluation(datasetId);
      let run = await response.json();
      setSelectedRun(run);
      for (let attempt = 0; ['queued', 'running'].includes(run.status) && attempt < 240; attempt += 1) {
        await sleep(2000);
        run = await fetchEvaluationRun(run.run_id);
        setSelectedRun(run);
      }
      setMessage(run.status === 'succeeded' ? '评测完成' : `评测状态：${run.status}`);
      await loadRuns(datasetId);
    } catch (error) { setMessage(`评测失败：${error.message}`); }
    finally { setBusy(false); }
  }

  async function startCaseRun(caseId) {
    if (!datasetId || !caseId || busy) return;
    setBusy(true);
    setRunningCaseId(caseId);
    setMessage('单条评测已提交，不计入完整数据集发布门禁');
    try {
      const response = await runEvaluationCase(datasetId, caseId);
      let run = await response.json();
      setSelectedRun(run);
      for (let attempt = 0; ['queued', 'running'].includes(run.status) && attempt < 240; attempt += 1) {
        await sleep(2000);
        run = await fetchEvaluationRun(run.run_id);
        setSelectedRun(run);
      }
      const casePassed = run.gate_result?.case_passed;
      setMessage(run.status === 'succeeded' ? `单条评测完成：${casePassed ? '通过' : '未通过'}` : `单条评测状态：${run.status}`);
      await loadRuns(datasetId);
    } catch (error) { setMessage(`单条评测失败：${error.message}`); }
    finally {
      setBusy(false);
      setRunningCaseId('');
    }
  }

  async function publishDataset() {
    if (!datasetId) return;
    try {
      await publishEvaluationDataset(datasetId);
      setMessage('数据集已发布并锁定');
      await loadDatasets();
    } catch (error) { setMessage(`发布失败：${error.message}`); }
  }

  async function loadCandidates() {
    try { setCandidates(await fetchEvaluationCandidates()); }
    catch (error) { setMessage(`候选样本读取失败：${error.message}`); }
  }

  async function mineCandidates() {
    setBusy(true);
    try {
      const created = await mineEvaluationCandidates();
      setMessage(`新增 ${created.length} 条候选样本`);
      await loadCandidates();
    } catch (error) { setMessage(`挖掘失败：${error.message}`); }
    finally { setBusy(false); }
  }

  async function labelCandidate(candidate, attributionCategory) {
    if (busy) return;
    setBusy(true);
    try {
      await reviewEvaluationCandidate(candidate.candidate_id, {
        status: 'labeled',
        severity: candidate.severity || 'medium',
        labels: ['production_failure', attributionCategory],
        attribution_category: attributionCategory,
        comment: `人工归因：${attributionLabel(attributionCategory)}`,
      });
      setMessage(`归因已保存：${attributionLabel(attributionCategory)}`);
      await loadCandidates();
    } catch (error) {
      setMessage(`审核标注失败：${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function promoteCandidate(candidate, expected) {
    if (!datasetId || busy) return;
    setBusy(true);
    try {
      const created = await promoteEvaluationCandidate(candidate.candidate_id, datasetId, expected);
      setPromotionCandidate(null);
      await Promise.all([loadCandidates(), loadCases(datasetId)]);
      setHighlightedCaseId(created.case_id);
      setTab('runs');
      setWorkspaceView('cases');
      setMessage(`回归用例已加入当前数据集：${created.case_id}`);
    } catch (error) {
      setMessage(`提升失败：${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  function openPromotionDialog(candidate) {
    const selectedDataset = datasets.find((dataset) => dataset.dataset_id === datasetId);
    if (!selectedDataset) {
      setMessage('请先选择一个评测数据集');
      return;
    }
    if (selectedDataset.status !== 'draft') {
      setMessage('当前数据集已发布并锁定，请选择 draft 状态的数据集后再提升用例');
      return;
    }
    setPromotionCandidate(candidate);
  }

  return (
    <section className={`evaluation-pane ${tab === 'runs' ? 'has-workspace-tabs' : ''}`}>
      <header className="chat-header evaluation-header">
        <div className="chat-heading">
          {sidebarCollapsed && <button className="icon-button" onClick={onToggleSidebar} title="展开侧栏"><PanelLeftOpen size={16} /></button>}
          <div><span className="page-kicker">质量控制</span><h2>评测中心</h2></div>
        </div>
        <button className="icon-button" onClick={() => { loadDatasets(); loadCandidates(); }} title="刷新评测数据"><RefreshCw size={15} /></button>
      </header>

      <div className="evaluation-toolbar">
        <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)} aria-label="评测数据集">
          <option value="">选择数据集</option>
          {datasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.name} · v{dataset.version} · {dataset.status}</option>)}
        </select>
        <div className="evaluation-actions">
          <button onClick={startRun} disabled={!datasetId || busy}><Play size={14} />运行全部</button>
          <button onClick={publishDataset} disabled={!datasetId}>发布数据集</button>
        </div>
      </div>

      <nav className="evaluation-tabs">
        <button className={tab === 'runs' ? 'active' : ''} onClick={() => setTab('runs')}>运行与门禁</button>
        <button className={tab === 'candidates' ? 'active' : ''} onClick={() => setTab('candidates')}>失败样本池 <span>{candidates.length}</span></button>
      </nav>

      {tab === 'runs' && (
        <nav className="evaluation-workspace-tabs" aria-label="评测工作区">
          <button className={workspaceView === 'cases' ? 'active' : ''} onClick={() => setWorkspaceView('cases')}>数据集样本 <span>{cases.length}</span></button>
          <button className={workspaceView === 'runs' ? 'active' : ''} onClick={() => setWorkspaceView('runs')}>评测运行 <span>{runs.length}</span></button>
        </nav>
      )}

      <div className={`evaluation-content ${tab === 'runs' && workspaceView === 'cases' ? 'cases-mode' : ''}`}>
        {tab === 'runs' ? (
          workspaceView === 'cases' ? (
            <DatasetCasesWorkspace cases={cases} runs={runs} onRunCase={startCaseRun} onOpenRun={openRunResult} busy={busy} runningCaseId={runningCaseId} highlightedCaseId={highlightedCaseId} />
          ) : (
            <RunWorkspace runs={runs} run={selectedRun} onOpenRun={openRun} onRunCase={startCaseRun} busy={busy} runningCaseId={runningCaseId} annotatingItemId={annotatingItemId} onAnnotate={annotateItem} />
          )
        ) : (
          <CandidateWorkspace candidates={candidates} onMine={mineCandidates} onLabel={labelCandidate} onPromote={openPromotionDialog} busy={busy} />
        )}
      </div>
      {message && <div className="evaluation-status">{message}</div>}
      {promotionCandidate && (
        <PromotionCaseDialog
          candidate={promotionCandidate}
          datasetId={datasetId}
          busy={busy}
          onClose={() => setPromotionCandidate(null)}
          onSubmit={(expected) => promoteCandidate(promotionCandidate, expected)}
        />
      )}
    </section>
  );
}

function DatasetCasesWorkspace({ cases, runs, onRunCase, onOpenRun, busy, runningCaseId, highlightedCaseId }) {
  const latestRuns = latestSingleRunsByCase(runs);
  const [expandedCaseId, setExpandedCaseId] = useState(highlightedCaseId || '');
  useEffect(() => {
    if (!highlightedCaseId) return;
    setExpandedCaseId(highlightedCaseId);
    const row = document.getElementById(`dataset-case-${highlightedCaseId}`);
    row?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [highlightedCaseId, cases]);
  return (
    <main className="dataset-case-workspace">
      <header className="dataset-case-heading">
        <div>
          <span>BASE CASES</span>
          <h3>数据集样本</h3>
          <p>这里展示所选数据集的全部样本定义；单条运行只会新增一条评测记录，不会改变此列表。</p>
        </div>
        <strong>{cases.length}<small> 条样本</small></strong>
      </header>
      <section className="dataset-case-list">
        {cases.map((item, index) => {
          const latestRun = latestRuns.get(item.case_id);
          const runState = singleRunState(latestRun);
          const running = runningCaseId === item.case_id;
          return (
            <article id={`dataset-case-${item.case_id}`} key={item.case_id} className={`dataset-case-row${item.case_id === highlightedCaseId ? ' is-newly-promoted' : ''}`}>
              <span className="dataset-case-index">{String(index + 1).padStart(3, '0')}</span>
              <div className="dataset-case-copy">
                <strong>{item.query}</strong>
                <div className="dataset-case-facts">
                  <span>{item.case_type || 'rag'}</span>
                  <span>{item.severity || 'medium'}</span>
                  {item.expected_intent && <span>意图 · {item.expected_intent}</span>}
                  <span className={item.reference_answer ? 'has-reference' : 'missing-reference'}>{item.reference_answer ? '有 Reference Answer' : '无 Reference Answer'}</span>
                </div>
              </div>
              <span className={`dataset-case-run-state ${running ? 'running' : runState.className}`}>{running ? '运行中' : runState.label}</span>
              <div className="dataset-case-actions">
                {latestRun && <button onClick={() => onOpenRun(latestRun.run_id)} disabled={busy}>查看结果</button>}
                <button className="evaluation-run-case" onClick={() => onRunCase(item.case_id)} disabled={busy || !item.enabled}>
                  <Play size={13} />{running ? '运行中…' : '单条运行'}
                </button>
              </div>
              <details className="dataset-case-reference" open={expandedCaseId === item.case_id} onToggle={(event) => { if (event.currentTarget.open) setExpandedCaseId(item.case_id); else if (expandedCaseId === item.case_id) setExpandedCaseId(''); }}>
                <summary>查看完整 Base Case</summary>
                <div className="dataset-case-reference-grid">
                  <BaseCaseField label="Case ID" value={item.case_id} code />
                  <BaseCaseField label="用例类型" value={item.case_type} />
                  <BaseCaseField label="严重级别" value={item.severity} />
                  <BaseCaseField label="启用状态" value={item.enabled ? '已启用' : '已停用'} />
                  <BaseCaseField label="期望意图" value={item.expected_intent} />
                  <BaseCaseField label="期望拒答" value={item.expected_refused == null ? '' : item.expected_refused ? '是' : '否'} />
                  <BaseCaseField label="地区编码" value={item.region_code} code />
                  <BaseCaseField label="政策版本" value={item.policy_version} />
                  <BaseCaseField label="业务类型" value={item.biz_type} />
                  <BaseCaseField label="申请编号" value={item.apply_id} code />
                  <BaseCaseField label="期望文档 ID" value={item.expected_doc_ids} code />
                  <BaseCaseField label="期望片段 ID" value={item.expected_chunk_ids} code />
                  <BaseCaseField label="期望答案关键词" value={item.expected_answer_keywords} />
                  <BaseCaseField label="标签" value={item.tags} />
                  <BaseCaseField label="Reference Answer" value={item.reference_answer} wide />
                  <BaseCaseField label="参考上下文" value={item.reference_contexts} wide />
                  <BaseCaseField label="期望工具轨迹" value={item.expected_trajectory} json wide />
                  <BaseCaseField label="期望业务结果" value={item.expected_outcome} json wide />
                  <BaseCaseField label="多轮对话" value={item.turns} json wide />
                  <BaseCaseField label="申请上下文" value={item.apply_context} json wide />
                  <BaseCaseField label="Agent 检索 Query" value={item.agent_retrieval_queries} json wide />
                  <BaseCaseField label="来源 Trace ID" value={item.source_trace_id} code />
                  <BaseCaseField label="来源元数据" value={item.source_metadata} json wide />
                  <BaseCaseField label="Fixture Key" value={item.fixture_key} code />
                  <BaseCaseField label="创建时间" value={item.created_at} />
                </div>
              </details>
            </article>
          );
        })}
        {!cases.length && <p className="hint">当前数据集没有可展示的样本。</p>}
      </section>
    </main>
  );
}

function BaseCaseField({ label, value, code = false, json = false, wide = false }) {
  const hasValue = Array.isArray(value)
    ? value.length > 0
    : value && typeof value === 'object'
      ? Object.keys(value).length > 0
      : value !== undefined && value !== null && String(value).trim() !== '';
  let content = '未配置';
  if (hasValue) {
    if (json) content = JSON.stringify(value, null, 2);
    else if (Array.isArray(value)) content = value.join(' · ');
    else content = String(value);
  }
  return (
    <div className={`base-case-field${wide ? ' is-wide' : ''}${!hasValue ? ' is-empty' : ''}`}>
      <span>{label}</span>
      {json ? <pre>{content}</pre> : <p>{code && hasValue ? <code>{content}</code> : content}</p>}
    </div>
  );
}

function latestSingleRunsByCase(runs) {
  const values = new Map();
  runs.forEach((run) => {
    const evaluation = run.runtime_snapshot?.evaluation || {};
    if (evaluation.scope === 'single_case' && evaluation.case_id && !values.has(evaluation.case_id)) values.set(evaluation.case_id, run);
  });
  return values;
}

function singleRunState(run) {
  if (!run) return { label: '未单独评测', className: 'idle' };
  if (['queued', 'running'].includes(run.status)) return { label: '运行中', className: 'running' };
  if (run.status === 'failed') return { label: '运行失败', className: 'failed' };
  return run.gate_result?.case_passed
    ? { label: '最近通过', className: 'passed' }
    : { label: '最近未通过', className: 'failed' };
}

function RunWorkspace({ runs, run, onOpenRun, onRunCase, onAnnotate, busy, runningCaseId, annotatingItemId }) {
  const gate = run?.gate_result || {};
  const metrics = run?.metrics || {};
  const singleCase = run?.runtime_snapshot?.evaluation?.scope === 'single_case' || gate.scope === 'single_case';
  const casePassed = gate.case_passed ?? run?.items?.[0]?.passed;
  return (
    <>
      <aside className="evaluation-run-list">
        {runs.map((item) => (
          <button key={item.run_id} className={run?.run_id === item.run_id ? 'active' : ''} onClick={() => onOpenRun(item.run_id)}>
            <strong>{item.runtime_snapshot?.evaluation?.scope === 'single_case' ? '单条' : '全量'} · {item.status}</strong><span>{item.run_id.slice(-10)}</span><small>{item.created_at}</small>
          </button>
        ))}
        {!runs.length && <p className="hint">暂无评测运行</p>}
      </aside>
      <main className="evaluation-detail">
        {!run ? <p className="hint">选择或启动一次评测运行。</p> : (
          <>
            <section className="evaluation-summary">
              <div><span>{singleCase ? '单条结果' : '门禁'}</span><strong className={(singleCase ? casePassed : gate.passed) ? 'pass' : 'fail'}>{(singleCase ? casePassed : gate.passed) ? <CheckCircle2 size={16} /> : <ShieldAlert size={16} />}{(singleCase ? casePassed : gate.passed) ? '通过' : '未通过'}</strong></div>
              <div><span>任务成功率</span><strong>{formatMetric(metrics.task_success_rate)}</strong></div>
              <div><span>用例</span><strong>{metrics.case_count ?? '-'}</strong></div>
              <div><span>P95延迟</span><strong>{metrics.performance?.p95_latency_ms ?? '-'} ms</strong></div>
            </section>
            {singleCase && <div className="evaluation-scope-note"><Play size={14} />单条调试运行仅验证当前用例，不计入完整数据集发布门禁。</div>}
            <EvaluationMetricGroups metrics={metrics} />
            {!singleCase && (gate.hard_failures?.length || gate.soft_failures?.length) > 0 && <div className="gate-failures">{[...(gate.hard_failures || []), ...(gate.soft_failures || [])].join(' · ')}</div>}
            <section className="evaluation-items">
              {(run.items || []).map((item) => <EvaluationItem key={item.item_id} item={item} onRunCase={onRunCase} onAnnotate={onAnnotate} busy={busy} running={runningCaseId === item.case_id} annotating={annotatingItemId === item.item_id} />)}
            </section>
          </>
        )}
      </main>
    </>
  );
}

function EvaluationMetricGroups({ metrics }) {
  const ragas = metrics.ragas || {};
  const ragasScores = Object.entries(ragas.scores || {});
  const judgeScores = Object.entries(metrics.judge || {});
  return (
    <section className="evaluation-metric-groups">
      <MetricGroup title="RAGAS" status={ragas.enabled ? `${ragas.evaluated_case_count || 0} 条已评分` : '本次未启用'} values={ragasScores} emptyText={ragas.enabled ? '没有可计算的 RAGAS 分数' : '运行时未启用 RAGAS'} />
      <MetricGroup title="LLM Judge" status={judgeScores.length ? `${judgeScores.length} 项指标` : '无评分'} values={judgeScores} emptyText="本次没有 Judge 分数" />
    </section>
  );
}

function MetricGroup({ title, status, values, emptyText }) {
  return (
    <div className="evaluation-metric-group">
      <header><strong>{title}</strong><span>{status}</span></header>
      {values.length ? (
        <div className="evaluation-metric-values">
          {values.map(([name, value]) => <div key={name}><span>{metricLabel(name)}</span><strong>{formatMetric(value)}</strong></div>)}
        </div>
      ) : <p>{emptyText}</p>}
    </div>
  );
}

function metricLabel(name) {
  return ({
    faithfulness: '忠实度',
    answer_relevancy: '答案相关性',
    context_precision: '上下文精确率',
    context_recall: '上下文召回率',
    completeness: '完整性',
    citation_precision: '引用精确率',
    citation_coverage: '引用覆盖率',
  })[name] || name;
}

function EvaluationItem({ item, onRunCase, onAnnotate, busy, running, annotating }) {
  const calls = item.actual_trajectory?.calls || [];
  const annotation = item.annotation_label || '';
  return (
    <details className={`evaluation-item ${item.passed ? 'pass' : 'fail'}`}>
      <summary>{item.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}<strong>{item.query}</strong><span>{calls.length} 次工具调用</span></summary>
      <div className="evaluation-item-body">
        <p>{item.answer || '无最终回答'}</p>
        {item.reference_answer && <div className="evaluation-reference-answer"><span>Reference Answer</span><p>{item.reference_answer}</p></div>}
        {!!Object.keys(item.ragas_scores || {}).length && <div className="evaluation-item-metrics">{Object.entries(item.ragas_scores).map(([name, value]) => <span key={name}>{metricLabel(name)} <strong>{formatMetric(value)}</strong></span>)}</div>}
        {!!item.failure_reasons?.length && <div className="failure-list">{item.failure_reasons.join(' · ')}</div>}
        <div className="trajectory-row">{calls.map((call, index) => <code key={`${call.tool_name}-${index}`}>{call.tool_name} · {call.status}</code>)}</div>
        <div className="evaluation-item-actions">
          <button className="evaluation-run-case" onClick={() => onRunCase(item.case_id)} disabled={busy || annotating || !item.case_id}><Play size={13} />{running ? '运行中…' : '单条运行'}</button>
          <button className={`evaluation-annotation pass ${annotation === 'pass' ? 'active' : ''}`} aria-pressed={annotation === 'pass'} onClick={() => onAnnotate(item.item_id, 'pass')} disabled={busy || annotating}>
            <CheckCircle2 size={13} />{annotating ? '保存中…' : annotation === 'pass' ? '已标记通过' : '标记通过'}
          </button>
          <button className={`evaluation-annotation fail ${annotation === 'fail' ? 'active' : ''}`} aria-pressed={annotation === 'fail'} onClick={() => onAnnotate(item.item_id, 'fail')} disabled={busy || annotating}>
            <XCircle size={13} />{annotating ? '保存中…' : annotation === 'fail' ? '已标记失败' : '标记失败'}
          </button>
          {annotation && <span className={`evaluation-annotation-state ${annotation}`}>人工标注 · {annotation === 'pass' ? '通过' : '失败'}</span>}
        </div>
      </div>
    </details>
  );
}

function CandidateWorkspace({ candidates, onMine, onLabel, onPromote, busy }) {
  const [reviewCategories, setReviewCategories] = useState({});
  function reviewCategoryFor(candidate) {
    return reviewCategories[candidate.candidate_id] || candidate.attribution_category || 'unknown';
  }
  return (
    <main className="candidate-workspace">
      <div className="candidate-toolbar"><button onClick={onMine} disabled={busy}><Search size={14} />扫描线上失败</button></div>
      {candidates.map((candidate) => {
        const reviewCategory = reviewCategoryFor(candidate);
        return (
          <article key={candidate.candidate_id} className="candidate-row">
            <div className="candidate-copy">
              <strong>{candidate.sanitized_query || '无问题文本'}</strong>
              <p>{candidate.sanitized_answer || '无回答文本'}</p>
              {candidate.attribution_reason && <small>{candidate.attribution_reason}</small>}
            </div>
            <div className="candidate-facts">
              <span>{candidate.status}</span><span>{candidate.severity}</span><span>{candidate.source}</span>
              <span className="is-attribution">{attributionLabel(candidate.attribution_category)}</span>
            </div>
            <div className="candidate-actions">
              <label className="candidate-attribution-select">
                <span>根因归因</span>
                <select
                  value={reviewCategory}
                  onChange={(event) => setReviewCategories((prev) => ({ ...prev, [candidate.candidate_id]: event.target.value }))}
                  disabled={candidate.status === 'promoted' || busy}
                  aria-label="根因归因"
                >
                  {ATTRIBUTION_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
              </label>
              <button onClick={() => onLabel(candidate, reviewCategory)} disabled={candidate.status === 'promoted' || busy}>保存归因</button>
              <button className="evaluation-run-case" onClick={() => onPromote(candidate)} disabled={candidate.status !== 'labeled' || busy}><ClipboardCheck size={14} />提升用例</button>
            </div>
          </article>
        );
      })}
      {!candidates.length && <p className="hint">暂无待处理失败样本</p>}
    </main>
  );
}

function PromotionCaseDialog({ candidate, datasetId, busy, onClose, onSubmit }) {
  const [form, setForm] = useState(() => promotionFormFromCandidate(candidate));
  const [error, setError] = useState('');
  const attributionCategory = candidate.attribution_category || 'unknown';

  useEffect(() => {
    setForm(promotionFormFromCandidate(candidate));
    setError('');
  }, [candidate.candidate_id]);

  useEffect(() => {
    function closeOnEscape(event) {
      if (event.key === 'Escape' && !busy) onClose();
    }
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [busy, onClose]);

  function update(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function parseObject(name, label) {
    const raw = String(form[name] || '').trim();
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') throw new Error(`${label}必须是 JSON 对象`);
    return parsed;
  }

  function submit(event) {
    event.preventDefault();
    try {
      const expectedTrajectory = parseObject('expected_trajectory', '期望工具轨迹');
      const expectedOutcome = parseObject('expected_outcome', '期望业务结果');
      const expected = {
        query: form.query.trim(),
        expected_intent: form.expected_intent.trim(),
        expected_refused: form.expected_refused === '' ? null : form.expected_refused === 'true',
        expected_doc_ids: splitFormList(form.expected_doc_ids),
        expected_chunk_ids: splitFormList(form.expected_chunk_ids),
        expected_answer_keywords: splitFormList(form.expected_answer_keywords),
        reference_answer: form.reference_answer.trim(),
        reference_contexts: splitFormList(form.reference_contexts),
        case_type: form.case_type,
        expected_trajectory: expectedTrajectory,
        expected_outcome: expectedOutcome,
        tags: Array.from(new Set([...splitFormList(form.tags), attributionCategory])),
        severity: form.severity,
        source_metadata: {
          candidate_id: candidate.candidate_id,
          attribution_category: attributionCategory,
        },
      };
      if (!expected.query) throw new Error('问题不能为空');
      if (attributionCategory === 'intent_mismatch' && !expected.expected_intent) throw new Error('意图识别错误必须填写期望意图');
      if (['retrieval_miss', 'knowledge_missing', 'stale_policy'].includes(attributionCategory)) {
        if (!expected.expected_doc_ids.length && !expected.expected_chunk_ids.length) throw new Error('该归因至少需要一个期望文档 ID 或片段 ID');
        if (!expected.reference_answer && !expected.expected_answer_keywords.length) throw new Error('该归因需要参考答案或期望答案关键词');
      }
      if (attributionCategory === 'hallucination' && !expected.reference_answer) throw new Error('事实幻觉用例必须填写经核实的参考答案');
      if (['tool_error', 'slot_error'].includes(attributionCategory) && !Object.keys(expectedOutcome).length && !(expectedTrajectory.required_tools || []).length) {
        throw new Error('该归因需要配置期望业务结果或必需工具');
      }
      setError('');
      onSubmit(expected);
    } catch (submitError) {
      setError(submitError.message || '请检查表单内容');
    }
  }

  return (
    <div className="promotion-dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onClose(); }}>
      <section className="promotion-dialog" role="dialog" aria-modal="true" aria-labelledby="promotion-dialog-title">
        <header>
          <div><span>回归基准</span><h3 id="promotion-dialog-title">提升为 Base Case</h3></div>
          <button className="icon-button" type="button" onClick={onClose} disabled={busy} title="关闭" aria-label="关闭"><X size={17} /></button>
        </header>
        <div className="promotion-dialog-context"><span>{attributionLabel(attributionCategory)}</span><code>{datasetId}</code></div>
        <form onSubmit={submit}>
          <fieldset>
            <legend>基础预期</legend>
            <div className="promotion-form-grid">
              <label className="is-wide">问题<input value={form.query} onChange={(event) => update('query', event.target.value)} maxLength="1000" /></label>
              <label>期望意图<input value={form.expected_intent} onChange={(event) => update('expected_intent', event.target.value)} placeholder="knowledge" maxLength="40" /></label>
              <label>期望拒答<select value={form.expected_refused} onChange={(event) => update('expected_refused', event.target.value)}><option value="">不校验</option><option value="false">正常回答</option><option value="true">应当拒答</option></select></label>
              <label>用例类型<select value={form.case_type} onChange={(event) => update('case_type', event.target.value)}><option value="agent">Agent</option><option value="rag">RAG</option><option value="multi_turn">多轮对话</option><option value="safety">安全拒答</option></select></label>
              <label>严重级别<select value={form.severity} onChange={(event) => update('severity', event.target.value)}><option value="low">低</option><option value="medium">中</option><option value="high">高</option><option value="critical">严重</option></select></label>
            </div>
          </fieldset>
          <fieldset>
            <legend>检索与答案基准</legend>
            <div className="promotion-form-grid">
              <label>期望文档 ID<textarea rows="3" value={form.expected_doc_ids} onChange={(event) => update('expected_doc_ids', event.target.value)} placeholder="doc-water-permit-guide" /></label>
              <label>期望片段 ID<textarea rows="3" value={form.expected_chunk_ids} onChange={(event) => update('expected_chunk_ids', event.target.value)} placeholder="chunk-water-permit-materials" /></label>
              <label>答案关键词<textarea rows="3" value={form.expected_answer_keywords} onChange={(event) => update('expected_answer_keywords', event.target.value)} placeholder="水资源论证报告&#10;取水许可申请书" /></label>
              <label>参考上下文<textarea rows="3" value={form.reference_contexts} onChange={(event) => update('reference_contexts', event.target.value)} placeholder="每行一段已核实上下文" /></label>
              <label className="is-wide">Reference Answer<textarea rows="4" value={form.reference_answer} onChange={(event) => update('reference_answer', event.target.value)} placeholder="填写经核实的标准答案" maxLength="20000" /></label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Agent 行为基准</legend>
            <div className="promotion-form-grid">
              <label>期望工具轨迹<textarea className="is-code" rows="8" value={form.expected_trajectory} onChange={(event) => update('expected_trajectory', event.target.value)} spellCheck="false" /></label>
              <label>期望业务结果<textarea className="is-code" rows="8" value={form.expected_outcome} onChange={(event) => update('expected_outcome', event.target.value)} spellCheck="false" /></label>
              <label className="is-wide">标签<input value={form.tags} onChange={(event) => update('tags', event.target.value)} placeholder="water_permit, retrieval_miss" /></label>
            </div>
          </fieldset>
          {error && <div className="promotion-form-error">{error}</div>}
          <footer><button type="button" onClick={onClose} disabled={busy}>取消</button><button className="primary" type="submit" disabled={busy}><ClipboardCheck size={15} />{busy ? '正在提升…' : '创建回归用例'}</button></footer>
        </form>
      </section>
    </div>
  );
}
