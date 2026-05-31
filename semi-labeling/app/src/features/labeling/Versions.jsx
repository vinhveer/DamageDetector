import { useCallback, useEffect, useState } from 'react';
import { Button, EmptyState, ErrorMessage, SelectControl, TextInput } from '../../components/ui/index.js';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

// Read-only management view: lists human labeling sessions and self-training
// (filter) rounds for a run, each tagged with name + timestamp.
export default function Versions({ dbPath, onChangeDbPath }) {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [sessions, setSessions] = useState([]);
  const [selftrain, setSelftrain] = useState([]);
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

  const loadAll = useCallback(async () => {
    if (!runId) return;
    setError('');
    try {
      const [s, t] = await Promise.all([
        api().listSessions({ resemiDbPath: dbPath, runId }),
        api().listSelfTrainingRuns({ resemiDbPath: dbPath, runId }),
      ]);
      setSessions(s.sessions || []);
      setSelftrain(t.runs || []);
    } catch (e) { setError(String(e?.message || e)); }
  }, [dbPath, runId]);

  useEffect(() => { loadRuns(); }, [loadRuns]);
  useEffect(() => { if (runId) loadAll(); }, [runId, loadAll]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <div>
          <div className="text-[15px] font-semibold text-[var(--text)]">Phiên bản</div>
          <div className="text-[13px] text-[var(--text-muted)]">Lịch sử label tay và các vòng lọc.</div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <TextInput
            value={dbPath}
            onChange={(e) => onChangeDbPath?.(e.currentTarget.value)}
            className="w-[320px]"
            placeholder="resemi.sqlite3"
          />
          {runs.length > 0 && (
            <SelectControl value={runId} onChange={(e) => setRunId(e.currentTarget.value)}>
              {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
            </SelectControl>
          )}
          <Button onClick={loadRuns}>Load</Button>
          <Button onClick={loadAll}>Refresh</Button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-2 gap-4 overflow-auto p-5">
        {error && <div className="col-span-2"><ErrorMessage>{error}</ErrorMessage></div>}

        <section className="flex min-h-0 flex-col">
          <div className="mb-2 text-[12px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            Phiên label tay ({sessions.length})
          </div>
          {sessions.length === 0 ? (
            <EmptyState title="Chưa có phiên label" />
          ) : (
            <div className="grid gap-2">
              {sessions.map((s) => (
                <div key={s.review_session_id} className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3 text-[12px]">
                  <div className="font-mono text-[12px] text-[var(--text)]">{s.review_session_id}</div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[var(--text-muted)]">
                    <span>{s.decision_count} nhãn</span>
                    <span>status: {s.status}</span>
                    <span>{fmt(s.created_at_utc)}</span>
                    {s.reviewer ? <span>by {s.reviewer}</span> : null}
                  </div>
                  {s.notes ? <div className="mt-1 text-[var(--text-subtle)]">{s.notes}</div> : null}
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="flex min-h-0 flex-col">
          <div className="mb-2 text-[12px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            Vòng lọc / self-training ({selftrain.length})
          </div>
          {selftrain.length === 0 ? (
            <EmptyState title="Chưa có vòng lọc" />
          ) : (
            <div className="grid gap-2">
              {selftrain.map((t) => (
                <div key={t.self_training_run_id} className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3 text-[12px]">
                  <div className="font-mono text-[12px] text-[var(--text)]">{t.self_training_run_id}</div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[var(--text-muted)]">
                    <span className="text-[var(--accent)]">promote {t.promoted_count}</span>
                    <span>defer {t.deferred_count}</span>
                    <span>reject {t.rejected_count}</span>
                    <span>round {t.round_index}</span>
                    <span>{fmt(t.created_at_utc)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

const fmt = (iso) => {
  try { return new Date(iso).toLocaleString(); } catch { return String(iso || ''); }
};
