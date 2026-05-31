import { useCallback, useEffect, useState } from 'react';
import { Button, EmptyState, ErrorMessage, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';
import BoxImage from './BoxImage.jsx';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

// Read-only management view: lists human labeling sessions and self-training
// (filter) rounds; opening one shows its contents with a box preview.
export default function Versions({ dbPath, onChangeDbPath, dataVersion }) {
  const [imageRoot, setImageRoot] = useState('');
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [sessions, setSessions] = useState([]);
  const [selftrain, setSelftrain] = useState([]);
  const [error, setError] = useState('');

  // detail: { kind: 'session'|'selftrain', id, rows }
  const [detail, setDetail] = useState(null);
  const [stActionFilter, setStActionFilter] = useState('all');
  const [selected, setSelected] = useState(null); // a row with box for preview

  useEffect(() => {
    const a = api();
    if (!a) return;
    a.getLabelingDefaults().then((d) => setImageRoot((p) => p || d.imageRootPath || '')).catch(() => {});
  }, []);

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
  useEffect(() => { if (runId) loadAll(); /* refresh after a step run */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataVersion]);

  const openSession = useCallback(async (sessionId) => {
    setSelected(null);
    try {
      const res = await api().getSessionDecisions({ resemiDbPath: dbPath, runId, reviewSessionId: sessionId, imageRootPath: imageRoot });
      setDetail({ kind: 'session', id: sessionId, rows: res.decisions || [] });
    } catch (e) { setError(String(e?.message || e)); }
  }, [dbPath, runId, imageRoot]);

  const openSelfTrain = useCallback(async (stId, action = 'all') => {
    setSelected(null);
    try {
      const res = await api().getSelfTrainingPromotions({ resemiDbPath: dbPath, runId, selfTrainingRunId: stId, action, imageRootPath: imageRoot });
      setDetail({ kind: 'selftrain', id: stId, rows: res.promotions || [] });
    } catch (e) { setError(String(e?.message || e)); }
  }, [dbPath, runId, imageRoot]);

  // re-fetch self-train detail when the action filter changes
  useEffect(() => {
    if (detail?.kind === 'selftrain') openSelfTrain(detail.id, stActionFilter);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stActionFilter]);

  if (detail) {
    return (
      <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
        <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
          <Button onClick={() => { setDetail(null); setSelected(null); }}>← Danh sách</Button>
          <span className="font-mono text-[var(--text)]">{detail.id}</span>
          <span className="text-[var(--text-muted)]">{detail.rows.length} mục</span>
          {detail.kind === 'selftrain' && (
            <SelectControl value={stActionFilter} onChange={(e) => setStActionFilter(e.currentTarget.value)} className="ml-2">
              <option value="all">action: tất cả</option>
              <option value="promote_clean">promote_clean</option>
              <option value="defer_review">defer_review</option>
              <option value="reject_candidate">reject_candidate</option>
            </SelectControl>
          )}
        </div>
        {error && <div className="px-4 pt-2"><ErrorMessage>{error}</ErrorMessage></div>}
        <div className="flex min-h-0 flex-1">
          <div className="w-[420px] shrink-0 overflow-auto border-r border-[var(--border-muted)] p-3">
            {detail.rows.length === 0 ? (
              <EmptyState title={detail.kind === 'session' ? 'Phiên rỗng' : 'Không có mục'} />
            ) : (
              <div className="grid gap-1.5">
                {detail.rows.map((row) => (
                  <button
                    key={row.resultId}
                    type="button"
                    onClick={() => setSelected(row)}
                    className={cn(
                      'rounded-[5px] border px-2.5 py-1.5 text-left text-[12px]',
                      selected?.resultId === row.resultId
                        ? 'border-[var(--accent)] bg-[var(--active)]'
                        : 'border-[var(--border)] hover:bg-[var(--hover)]',
                    )}
                  >
                    <span className="text-[var(--text-muted)]">#{row.resultId}</span>{' '}
                    {detail.kind === 'session' ? (
                      <span className="text-[var(--text)]">{row.action}: {row.previousLabel || '—'} → {row.newLabel || '—'}</span>
                    ) : (
                      <span className="text-[var(--text)]">
                        {row.action} · {row.predictedLabel} ({(row.classifierConfidence * 100).toFixed(0)}%, m {row.classifierMargin.toFixed(2)})
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="min-w-0 flex-1 p-4">
            {selected ? (
              <BoxImage imageUri={selected.imageUri} cropUri={selected.cropUri} box={selected.box} />
            ) : (
              <div className="flex h-full items-center justify-center text-[13px] text-[var(--text-muted)]">Chọn một mục để xem ảnh.</div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <div>
          <div className="text-[15px] font-semibold text-[var(--text)]">Phiên bản</div>
          <div className="text-[13px] text-[var(--text-muted)]">Lịch sử label tay và các vòng lọc. Bấm để xem chi tiết.</div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <TextInput
            value={dbPath}
            onChange={(e) => onChangeDbPath?.(e.currentTarget.value)}
            className="w-[280px]"
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
                <button
                  key={s.review_session_id}
                  type="button"
                  onClick={() => openSession(s.review_session_id)}
                  className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3 text-left text-[12px] hover:border-[var(--text-subtle)] hover:bg-[var(--hover)]"
                >
                  <div className="font-mono text-[12px] text-[var(--text)]">{s.review_session_id}</div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[var(--text-muted)]">
                    <span>{s.decision_count} nhãn</span>
                    <span>status: {s.status}</span>
                    <span>{fmt(s.created_at_utc)}</span>
                    {s.reviewer ? <span>by {s.reviewer}</span> : null}
                  </div>
                  {s.notes ? <div className="mt-1 text-[var(--text-subtle)]">{s.notes}</div> : null}
                </button>
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
                <button
                  key={t.self_training_run_id}
                  type="button"
                  onClick={() => { setStActionFilter('all'); openSelfTrain(t.self_training_run_id, 'all'); }}
                  className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3 text-left text-[12px] hover:border-[var(--text-subtle)] hover:bg-[var(--hover)]"
                >
                  <div className="font-mono text-[12px] text-[var(--text)]">{t.self_training_run_id}</div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[var(--text-muted)]">
                    <span className="text-[var(--accent)]">promote {t.promoted_count}</span>
                    <span>defer {t.deferred_count}</span>
                    <span>reject {t.rejected_count}</span>
                    <span>round {t.round_index}</span>
                    <span>{fmt(t.created_at_utc)}</span>
                  </div>
                </button>
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
