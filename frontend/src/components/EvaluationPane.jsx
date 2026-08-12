import React, { useEffect, useState } from 'react';
import { CheckCircle2, PanelLeftOpen, Play, RefreshCw, Search, ShieldAlert, XCircle } from 'lucide-react';
import {
  annotateEvalItem,
  createEvaluationKnowledgeDraft,
  fetchEvaluationCandidates,
  fetchEvaluationCases,
  fetchEvaluationDatasets,
  fetchEvaluationRun,
  fetchEvaluationRuns,
  formatMetric,
  mineEvaluationCandidates,
  promoteEvaluationCandidate,
  publishEvaluationKnowledgeDraft,
  publishEvaluationDataset,
  reviewEvaluationCandidate,
  reviewEvaluationKnowledgeDraft,
  runEvaluation,
  runEvaluationCase,
  sleep,
} from '../utils.js';

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
  const [message, setMessage] = useState('');

  useEffect(() => { loadDatasets(); }, []);
  useEffect(() => {
    if (!datasetId) return;
    setCases([]);
    setRuns([]);
    setSelectedRun(null);
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

  async function labelCandidate(candidate) {
    await reviewEvaluationCandidate(candidate.candidate_id, {
      status: 'labeled', severity: candidate.severity || 'medium', labels: ['production_failure', candidate.attribution_category || 'unknown'], attribution_category: candidate.attribution_category || 'unknown', comment: '前端人工标注',
    });
    await loadCandidates();
  }

  async function promoteCandidate(candidate) {
    if (!datasetId) return;
    try {
      await promoteEvaluationCandidate(candidate.candidate_id, datasetId, {
        query: candidate.sanitized_query,
        case_type: 'agent', severity: candidate.severity || 'medium', tags: candidate.labels || [],
      });
      setMessage('候选样本已提升为回归用例');
      await loadCandidates();
    } catch (error) { setMessage(`提升失败：${error.message}`); }
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
            <DatasetCasesWorkspace cases={cases} runs={runs} onRunCase={startCaseRun} onOpenRun={openRunResult} busy={busy} runningCaseId={runningCaseId} />
          ) : (
            <RunWorkspace runs={runs} run={selectedRun} onOpenRun={openRun} onRunCase={startCaseRun} busy={busy} runningCaseId={runningCaseId} annotatingItemId={annotatingItemId} onAnnotate={annotateItem} />
          )
        ) : (
          <CandidateWorkspace candidates={candidates} onMine={mineCandidates} onLabel={labelCandidate} onPromote={promoteCandidate} onCreateKnowledgeDraft={async (candidate, payload) => { await createEvaluationKnowledgeDraft(candidate.candidate_id, payload); setMessage('知识修订草稿已保存'); await loadCandidates(); }} onReviewKnowledgeDraft={async (candidate, status) => { await reviewEvaluationKnowledgeDraft(candidate.candidate_id, { status, comment: '前端审核知识修订' }); setMessage(status === 'approved' ? '知识修订已审核通过' : '知识修订已驳回'); await loadCandidates(); }} onPublishKnowledgeDraft={async (candidate) => { await publishEvaluationKnowledgeDraft(candidate.candidate_id); setMessage('知识修订已发布并进入索引'); await loadCandidates(); }} busy={busy} />
        )}
      </div>
      {message && <div className="evaluation-status">{message}</div>}
    </section>
  );
}

function DatasetCasesWorkspace({ cases, runs, onRunCase, onOpenRun, busy, runningCaseId }) {
  const latestRuns = latestSingleRunsByCase(runs);
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
            <article key={item.case_id} className="dataset-case-row">
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
              <details className="dataset-case-reference">
                <summary>查看样本基准</summary>
                <div className="dataset-case-reference-grid">
                  <div><span>Reference Answer</span><p>{item.reference_answer || '未配置'}</p></div>
                  <div><span>期望关键词</span><p>{item.expected_answer_keywords?.length ? item.expected_answer_keywords.join(' · ') : '未配置'}</p></div>
                  <div><span>期望文档</span><p>{item.expected_doc_ids?.length ? item.expected_doc_ids.join(' · ') : '未配置'}</p></div>
                  <div><span>Case ID</span><p><code>{item.case_id}</code></p></div>
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

function CandidateWorkspace({ candidates, onMine, onLabel, onPromote, onCreateKnowledgeDraft, onReviewKnowledgeDraft, onPublishKnowledgeDraft, busy }) {
  const [drafts, setDrafts] = useState({});
  const categories = ['retrieval_miss', 'knowledge_missing', 'hallucination', 'stale_policy'];
  function draftFor(candidate) {
    return drafts[candidate.candidate_id] || candidate.knowledge_draft || {};
  }
  return (
    <main className="candidate-workspace">
      <div className="candidate-toolbar"><button onClick={onMine} disabled={busy}><Search size={14} />扫描线上失败</button></div>
      {candidates.map((candidate) => (
        <article key={candidate.candidate_id} className="candidate-row">
          <div><strong>{candidate.sanitized_query || '无问题文本'}</strong><p>{candidate.sanitized_answer || '无回答文本'}</p></div>
          <div className="candidate-facts"><span>{candidate.status}</span><span>{candidate.severity}</span><span>{candidate.source}</span><span>{candidate.attribution_category || 'unknown'}</span></div>
          <div className="candidate-actions"><button onClick={() => onLabel(candidate)} disabled={candidate.status === 'promoted'}>审核标注</button><button onClick={() => onPromote(candidate)} disabled={candidate.status !== 'labeled'}>提升用例</button></div>
          <div className="candidate-knowledge-actions">
            <select value={draftFor(candidate).category || (categories.includes(candidate.attribution_category) ? candidate.attribution_category : 'knowledge_missing')} onChange={(event) => setDrafts((prev) => ({ ...prev, [candidate.candidate_id]: { ...draftFor(candidate), category: event.target.value } }))} aria-label="知识归因类型">
              {categories.map((category) => <option key={category} value={category}>{category}</option>)}
            </select>
            <textarea rows="3" value={draftFor(candidate).content || ''} onChange={(event) => setDrafts((prev) => ({ ...prev, [candidate.candidate_id]: { ...draftFor(candidate), content: event.target.value } }))} placeholder="填写经核实的 FAQ 或政策内容" />
            <div><button onClick={() => onCreateKnowledgeDraft(candidate, { category: draftFor(candidate).category || 'knowledge_missing', title: draftFor(candidate).title || `${candidate.sanitized_query || 'Badcase'} 知识修订`, content: draftFor(candidate).content || '', region_code: '340000', policy_version: '2025', change_note: 'Badcase 知识修订' })} disabled={!String(draftFor(candidate).content || '').trim()}>保存知识草稿</button><button onClick={() => onReviewKnowledgeDraft(candidate, 'approved')} disabled={draftFor(candidate).status !== 'draft'}>审核通过</button><button onClick={() => onReviewKnowledgeDraft(candidate, 'rejected')} disabled={draftFor(candidate).status !== 'draft'}>驳回</button><button onClick={() => onPublishKnowledgeDraft(candidate)} disabled={draftFor(candidate).status !== 'approved'}>发布知识</button></div>
            {draftFor(candidate).status && <small>知识草稿：{draftFor(candidate).status}</small>}
          </div>
        </article>
      ))}
      {!candidates.length && <p className="hint">暂无待处理失败样本</p>}
    </main>
  );
}
