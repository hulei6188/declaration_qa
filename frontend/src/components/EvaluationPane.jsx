import React, { useEffect, useState } from 'react';
import { CheckCircle2, FlaskConical, MessageSquare, Play, RefreshCw, Search, ShieldAlert, XCircle } from 'lucide-react';
import {
  annotateEvalItem,
  fetchEvaluationCandidates,
  fetchEvaluationDatasets,
  fetchEvaluationRun,
  fetchEvaluationRuns,
  formatMetric,
  mineEvaluationCandidates,
  promoteEvaluationCandidate,
  publishEvaluationDataset,
  reviewEvaluationCandidate,
  runEvaluation,
  sleep,
} from '../utils.js';

export default function EvaluationPane({ onViewChange, sidebarCollapsed, onToggleSidebar }) {
  const [tab, setTab] = useState('runs');
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState('');
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => { loadDatasets(); }, []);
  useEffect(() => { if (datasetId) loadRuns(datasetId); }, [datasetId]);
  useEffect(() => { if (tab === 'candidates') loadCandidates(); }, [tab]);

  async function loadDatasets() {
    try {
      const values = await fetchEvaluationDatasets();
      setDatasets(values || []);
      setDatasetId((current) => current || values?.[0]?.dataset_id || '');
    } catch (error) { setMessage(`数据集读取失败：${error.message}`); }
  }

  async function loadRuns(id = datasetId) {
    if (!id) return;
    try {
      const values = await fetchEvaluationRuns(id);
      setRuns(values || []);
      if (values?.[0]?.run_id) await openRun(values[0].run_id);
    } catch (error) { setMessage(`运行记录读取失败：${error.message}`); }
  }

  async function openRun(runId) {
    try { setSelectedRun(await fetchEvaluationRun(runId)); }
    catch (error) { setMessage(`运行详情读取失败：${error.message}`); }
  }

  async function startRun() {
    if (!datasetId || busy) return;
    setBusy(true);
    setMessage('评测任务已提交');
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
      status: 'labeled', severity: candidate.severity || 'medium', labels: ['production_failure'], comment: '前端人工标注',
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
    <section className="evaluation-pane">
      <header className="chat-header evaluation-header">
        <div className="chat-heading">
          {sidebarCollapsed && <button className="icon-button" onClick={onToggleSidebar} title="展开侧栏"><MessageSquare size={16} /></button>}
          <h2>Gov Service Agent</h2>
          <nav className="view-tabs" aria-label="工作区">
            <button onClick={() => onViewChange('chat')}><MessageSquare size={14} />对话</button>
            <button className="active"><FlaskConical size={14} />评测</button>
          </nav>
        </div>
        <button className="icon-button" onClick={() => { loadDatasets(); loadCandidates(); }} title="刷新评测数据"><RefreshCw size={15} /></button>
      </header>

      <div className="evaluation-toolbar">
        <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)} aria-label="评测数据集">
          <option value="">选择数据集</option>
          {datasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.name} · v{dataset.version} · {dataset.status}</option>)}
        </select>
        <div className="evaluation-actions">
          <button onClick={startRun} disabled={!datasetId || busy}><Play size={14} />运行</button>
          <button onClick={publishDataset} disabled={!datasetId}>发布数据集</button>
        </div>
      </div>

      <nav className="evaluation-tabs">
        <button className={tab === 'runs' ? 'active' : ''} onClick={() => setTab('runs')}>运行与门禁</button>
        <button className={tab === 'candidates' ? 'active' : ''} onClick={() => setTab('candidates')}>失败样本池 <span>{candidates.length}</span></button>
      </nav>

      <div className="evaluation-content">
        {tab === 'runs' ? (
          <RunWorkspace runs={runs} run={selectedRun} onOpenRun={openRun} onAnnotate={async (itemId, label) => {
            await annotateEvalItem(itemId, label); await openRun(selectedRun.run_id);
          }} />
        ) : (
          <CandidateWorkspace candidates={candidates} onMine={mineCandidates} onLabel={labelCandidate} onPromote={promoteCandidate} busy={busy} />
        )}
      </div>
      {message && <div className="evaluation-status">{message}</div>}
    </section>
  );
}

function RunWorkspace({ runs, run, onOpenRun, onAnnotate }) {
  const gate = run?.gate_result || {};
  const metrics = run?.metrics || {};
  return (
    <>
      <aside className="evaluation-run-list">
        {runs.map((item) => (
          <button key={item.run_id} className={run?.run_id === item.run_id ? 'active' : ''} onClick={() => onOpenRun(item.run_id)}>
            <strong>{item.status}</strong><span>{item.run_id.slice(-10)}</span><small>{item.created_at}</small>
          </button>
        ))}
        {!runs.length && <p className="hint">暂无评测运行</p>}
      </aside>
      <main className="evaluation-detail">
        {!run ? <p className="hint">选择或启动一次评测运行。</p> : (
          <>
            <section className="evaluation-summary">
              <div><span>门禁</span><strong className={gate.passed ? 'pass' : 'fail'}>{gate.passed ? <CheckCircle2 size={16} /> : <ShieldAlert size={16} />}{gate.passed ? '通过' : '未通过'}</strong></div>
              <div><span>任务成功率</span><strong>{formatMetric(metrics.task_success_rate)}</strong></div>
              <div><span>用例</span><strong>{metrics.case_count ?? '-'}</strong></div>
              <div><span>P95延迟</span><strong>{metrics.performance?.p95_latency_ms ?? '-'} ms</strong></div>
            </section>
            {(gate.hard_failures?.length || gate.soft_failures?.length) > 0 && <div className="gate-failures">{[...(gate.hard_failures || []), ...(gate.soft_failures || [])].join(' · ')}</div>}
            <section className="evaluation-items">
              {(run.items || []).map((item) => <EvaluationItem key={item.item_id} item={item} onAnnotate={onAnnotate} />)}
            </section>
          </>
        )}
      </main>
    </>
  );
}

function EvaluationItem({ item, onAnnotate }) {
  const calls = item.actual_trajectory?.calls || [];
  return (
    <details className={`evaluation-item ${item.passed ? 'pass' : 'fail'}`}>
      <summary>{item.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}<strong>{item.query}</strong><span>{calls.length} 次工具调用</span></summary>
      <div className="evaluation-item-body">
        <p>{item.answer || '无最终回答'}</p>
        {!!item.failure_reasons?.length && <div className="failure-list">{item.failure_reasons.join(' · ')}</div>}
        <div className="trajectory-row">{calls.map((call, index) => <code key={`${call.tool_name}-${index}`}>{call.tool_name} · {call.status}</code>)}</div>
        <div className="evaluation-item-actions"><button onClick={() => onAnnotate(item.item_id, 'pass')}>标记通过</button><button onClick={() => onAnnotate(item.item_id, 'fail')}>标记失败</button></div>
      </div>
    </details>
  );
}

function CandidateWorkspace({ candidates, onMine, onLabel, onPromote, busy }) {
  return (
    <main className="candidate-workspace">
      <div className="candidate-toolbar"><button onClick={onMine} disabled={busy}><Search size={14} />扫描线上失败</button></div>
      {candidates.map((candidate) => (
        <article key={candidate.candidate_id} className="candidate-row">
          <div><strong>{candidate.sanitized_query || '无问题文本'}</strong><p>{candidate.sanitized_answer || '无回答文本'}</p></div>
          <div className="candidate-facts"><span>{candidate.status}</span><span>{candidate.severity}</span><span>{candidate.source}</span></div>
          <div className="candidate-actions"><button onClick={() => onLabel(candidate)} disabled={candidate.status === 'promoted'}>审核标注</button><button onClick={() => onPromote(candidate)} disabled={candidate.status !== 'labeled'}>提升用例</button></div>
        </article>
      ))}
      {!candidates.length && <p className="hint">暂无待处理失败样本</p>}
    </main>
  );
}
