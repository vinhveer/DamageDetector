import { useCallback, useEffect, useState } from 'react';
import { Button, EmptyState, ErrorMessage, SelectControl, TextInput } from '../../components/ui/index.js';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

// R6: progress metrics across self-training rounds for a run.
export default function Metrics({ dbPath, onChangeDbPath, dataVersion }) {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState('');

  const loadRuns = useCallback(async () => {
    setError('');
    try {
      const res = await api().listLabelingRuns({ resemiDbPath: dbPath });
      const list = res.runs || [];
      setRuns(list);
      if (list.length && !runId) setRunId(list[0].run_id);
    } catch (e) { setError(String(e?.message || e)); }
  }, [dbPath, runId]);

  const load = useCallback(async () => {
    if (!runId) return;
    setError('');
    try {
      const res = await api().getRunMetrics({ resemiDbPath: dbPath, runId });
      setMetrics(res);
    } catch (e) { setError(String(e?.message || e)); }
  }, [dbPath, runId]);

  useEffect(() => { loadRuns(); }, [loadRuns]);
  useEffect(() => { if (runId) load(); }, [runId, load]);
  useEffect(() => { if (runId) load(); /* refresh after a step run */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataVersion]);

  const pct = (v) => (v == null ? 'N/A' : `${(v * 100).toFixed(1)}%`);

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <div>
          <div className="text-[15px] font-semibold text-[var(--text)]">Chỉ số</div>
          <div className="text-[13px] text-[var(--text-muted)]">Tiến triển qua các vòng self-training.</div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <TextInput value={dbPath} onChange={(e) => onChangeDbPath?.(e.currentTarget.value)} className="w-[280px]" placeholder="resemi.sqlite3" />
          {runs.length > 0 && (
            <SelectControl value={runId} onChange={(e) => setRunId(e.currentTarget.value)}>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
            </SelectControl>
          )}
          <Button onClick={loadRuns}>Load</Button>
          <Button onClick={load}>Refresh</Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-5">
        {error && <ErrorMessage>{error}</ErrorMessage>}

        {metrics && (
          <div className="mb-4 flex flex-wrap gap-4 text-[13px]">
            <Stat label="cleaned_labels" value={metrics.cleanedCount} />
            <Stat label="review_queue còn lại" value={metrics.reviewQueueCount} />
            <Stat label="số vòng" value={metrics.rounds.length} />
            {metrics.deferredTrend.length > 1 && <Sparkline values={metrics.deferredTrend} />}
          </div>
        )}

        {metrics && metrics.rounds.length === 0 ? (
          <EmptyState title="Chưa có vòng self-training" message="Chạy step08 + step09 ở tab Chạy bước." />
        ) : metrics ? (
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr className="border-b border-[var(--border)] text-left text-[var(--text-muted)]">
                <th className="py-2 pr-3">round</th>
                <th className="py-2 pr-3">self_training_run_id</th>
                <th className="py-2 pr-3 text-right">promote</th>
                <th className="py-2 pr-3 text-right">defer</th>
                <th className="py-2 pr-3 text-right">reject</th>
                <th className="py-2 pr-3 text-right">accuracy (CV)</th>
                <th className="py-2 pr-3 text-right">OOF disagree</th>
              </tr>
            </thead>
            <tbody>
              {metrics.rounds.map((r) => (
                <tr key={r.selfTrainingRunId} className="border-b border-[var(--border-muted)]">
                  <td className="py-2 pr-3 text-[var(--text)]">{r.roundIndex}</td>
                  <td className="py-2 pr-3 font-mono text-[11px] text-[var(--text-muted)]">{r.selfTrainingRunId}</td>
                  <td className="py-2 pr-3 text-right text-[var(--accent)]">{r.promotedCount}</td>
                  <td className="py-2 pr-3 text-right text-[var(--text)]">{r.deferredCount}</td>
                  <td className="py-2 pr-3 text-right text-[var(--text)]">{r.rejectedCount}</td>
                  <td className="py-2 pr-3 text-right text-[var(--text)]">{pct(r.classifierAccuracy)}</td>
                  <td className="py-2 pr-3 text-right text-[var(--text)]">{pct(r.oofDisagreementRatio)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-[13px] text-[var(--text-muted)]">Chọn run rồi Refresh.</div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)] px-3 py-2">
      <div className="text-[11px] text-[var(--text-muted)]">{label}</div>
      <div className="text-[16px] font-semibold text-[var(--text)]">{value}</div>
    </div>
  );
}

// Tiny inline SVG sparkline of the deferred-count trend.
function Sparkline({ values }) {
  const w = 160;
  const h = 40;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * (w - 4) + 2;
    const y = h - 2 - ((v - min) / span) * (h - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <div className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)] px-3 py-2">
      <div className="text-[11px] text-[var(--text-muted)]">defer theo vòng</div>
      <svg width={w} height={h} className="mt-1">
        <polyline points={pts} fill="none" stroke="var(--primary)" strokeWidth="1.5" />
      </svg>
    </div>
  );
}
