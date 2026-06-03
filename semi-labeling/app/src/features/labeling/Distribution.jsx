import { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, EmptyState, ErrorMessage, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);
const GRID_LIMIT = 500; // max thumbnails pulled per class

// "Phân bố" tab: per-class % of cleaned_labels + a scan grid to eyeball whether
// each crop is labelled correctly, with multi-select bulk relabel + commit.
export default function Distribution({ dbPath, onChangeDbPath, dataVersion }) {
  const [imageRoot, setImageRoot] = useState('');
  const [labels, setLabels] = useState(['crack', 'mold', 'spall', 'stain', 'reject']);
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [dist, setDist] = useState(null);
  const [selectedClass, setSelectedClass] = useState('');
  const [gridItems, setGridItems] = useState([]);
  const [selected, setSelected] = useState(() => new Set());
  const [pendingEdits, setPendingEdits] = useState({});
  const [sessionName, setSessionName] = useState('');
  const [loading, setLoading] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const a = api();
    if (!a) return;
    a.getLabelingDefaults().then((d) => {
      setImageRoot((p) => p || d.imageRootPath || '');
      if (Array.isArray(d.labels) && d.labels.length) setLabels(d.labels);
    }).catch(() => {});
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

  const loadDist = useCallback(async () => {
    if (!runId) { setError('Chọn run trước.'); return; }
    setLoading(true);
    setError('');
    try {
      const res = await api().cleanedDistribution({ resemiDbPath: dbPath, runId });
      setDist(res);
    } catch (e) { setError(String(e?.message || e)); } finally { setLoading(false); }
  }, [dbPath, runId]);

  const selectClass = useCallback(async (label) => {
    setSelectedClass(label);
    setSelected(new Set());
    setLoading(true);
    setError('');
    try {
      const res = await api().listCleaned({
        resemiDbPath: dbPath, runId, imageRootPath: imageRoot,
        finalLabel: label, decisionType: 'all', limit: GRID_LIMIT,
      });
      setGridItems(res.items || []);
    } catch (e) { setError(String(e?.message || e)); } finally { setLoading(false); }
  }, [dbPath, runId, imageRoot]);

  // refresh after a commit / step run
  useEffect(() => {
    if (runId) loadDist();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataVersion]);

  const toggleSel = useCallback((resultId) => {
    setSelected((prev) => {
      const n = new Set(prev);
      if (n.has(resultId)) n.delete(resultId); else n.add(resultId);
      return n;
    });
  }, []);

  const relabelSelected = useCallback(async (newLabel) => {
    if (selected.size === 0) return;
    setError('');
    const ids = [...selected];
    const edits = {};
    for (const rid of ids) {
      const it = gridItems.find((x) => x.resultId === rid);
      if (!it || it.finalLabel === newLabel) continue;
      try {
        const res = await api().updateCleanedLabel({ resemiDbPath: dbPath, runId, resultId: rid, newLabel });
        if (res.error) { setError(res.error); continue; }
        edits[rid] = {
          action: newLabel === 'reject' ? 'manual_reject' : 'manual_relabel',
          previousLabel: it.finalLabel,
          newLabel,
        };
      } catch (e) { setError(String(e?.message || e)); }
    }
    setGridItems((arr) => arr.map((it) => (selected.has(it.resultId) ? { ...it, finalLabel: newLabel } : it)));
    setPendingEdits((p) => ({ ...p, ...edits }));
    setSelected(new Set());
  }, [selected, gridItems, dbPath, runId]);

  const commit = useCallback(async () => {
    const edits = Object.entries(pendingEdits).map(([rid, d]) => ({
      resultId: Number(rid), action: d.action, previousLabel: d.previousLabel, newLabel: d.newLabel,
    }));
    if (!edits.length) { setError('Chưa có chỉnh sửa nào'); return; }
    setCommitting(true);
    setError('');
    try {
      const res = await api().commitCorrections({ resemiDbPath: dbPath, runId, sessionName, edits });
      if (res.error) { setError(res.error); return; }
      alert(`Đã ghi ${res.decisionCount} sửa vào session ${res.reviewSessionId}`);
      setPendingEdits({});
      await loadDist();
      if (selectedClass) await selectClass(selectedClass);
    } catch (e) { setError(String(e?.message || e)); } finally { setCommitting(false); }
  }, [pendingEdits, dbPath, runId, sessionName, selectedClass, loadDist, selectClass]);

  const pendingCount = Object.keys(pendingEdits).length;
  const maxCount = useMemo(
    () => Math.max(1, ...((dist?.byLabel || []).map((b) => b.count))),
    [dist],
  );
  const classCount = useMemo(
    () => dist?.byLabel?.find((b) => b.key === selectedClass)?.count ?? 0,
    [dist, selectedClass],
  );

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      {/* control bar */}
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
        <span className="text-[15px] font-semibold text-[var(--text)]">Phân bố</span>
        <TextInput value={dbPath} onChange={(e) => onChangeDbPath?.(e.currentTarget.value)} className="w-[260px]" placeholder="resemi.sqlite3" />
        <Button onClick={loadRuns}>Load runs</Button>
        {runs.length > 0 && (
          <SelectControl value={runId} onChange={(e) => setRunId(e.currentTarget.value)}>
            {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
          </SelectControl>
        )}
        <Button onClick={loadDist} disabled={!runId || loading}>{loading ? 'Loading…' : 'Xem phân bố'}</Button>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-[var(--text)]">đã sửa: {pendingCount}</span>
          <TextInput value={sessionName} onChange={(e) => setSessionName(e.currentTarget.value)} placeholder="tên phiên" className="w-[150px]" />
          <Button onClick={commit} disabled={committing || pendingCount === 0}>{committing ? 'Committing…' : `Commit (${pendingCount})`}</Button>
        </div>
      </div>

      {error && <div className="px-4 pt-2"><ErrorMessage>{error}</ErrorMessage></div>}

      {!dist ? (
        <div className="flex flex-1 items-center justify-center">
          <EmptyState title="Chưa có dữ liệu" message="Chọn run rồi bấm 'Xem phân bố'." />
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          {/* distribution panel */}
          <div className="shrink-0 border-b border-[var(--border-muted)] p-4">
            <div className="mb-2 text-[13px] text-[var(--text-muted)]">
              Tổng cleaned_labels: <span className="font-semibold text-[var(--text)]">{dist.total}</span> — bấm 1 lớp để soi ảnh
            </div>
            <div className="grid gap-1.5">
              {dist.byLabel.map((b) => (
                <button
                  key={b.key}
                  type="button"
                  onClick={() => selectClass(b.key)}
                  className={cn(
                    'group flex items-center gap-3 rounded-[5px] border px-2 py-1.5 text-left text-[12px]',
                    selectedClass === b.key ? 'border-[var(--accent)] bg-[var(--active)]' : 'border-[var(--border)] hover:bg-[var(--hover)]',
                  )}
                >
                  <span className="w-[64px] shrink-0 font-medium text-[var(--text)]">{b.key}</span>
                  <span className="relative h-[14px] flex-1 overflow-hidden rounded-[3px] bg-[var(--surface-2)]">
                    <span
                      className="absolute inset-y-0 left-0 rounded-[3px] bg-[var(--primary)]"
                      style={{ width: `${Math.max(2, (b.count / maxCount) * 100)}%` }}
                    />
                  </span>
                  <span className="w-[120px] shrink-0 text-right text-[var(--text-muted)]">
                    {b.count} <span className="text-[var(--text-subtle)]">({(b.pct * 100).toFixed(1)}%)</span>
                  </span>
                </button>
              ))}
            </div>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-[var(--text-subtle)]">
              {dist.byDecisionType.map((t) => (
                <span key={t.key} className="rounded-[4px] bg-[var(--surface-2)] px-1.5 py-0.5">
                  {t.key}: {t.count} ({(t.pct * 100).toFixed(1)}%)
                </span>
              ))}
            </div>
          </div>

          {/* scan grid */}
          {selectedClass ? (
            <div className="flex min-h-0 flex-1 flex-col">
              <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-1.5 text-[12px]">
                <span className="text-[var(--text)]">Lớp <b>{selectedClass}</b></span>
                <span className="text-[var(--text-subtle)]">hiện {gridItems.length} / {classCount}{classCount > GRID_LIMIT ? ` (cap ${GRID_LIMIT})` : ''}</span>
                <span className="text-[var(--text-muted)]">· chọn: {selected.size}</span>
                <div className="ml-auto flex items-center gap-1">
                  <span className="text-[var(--text-subtle)]">Đổi nhãn đã chọn →</span>
                  {labels.map((lab) => (
                    <button
                      key={lab}
                      type="button"
                      onClick={() => relabelSelected(lab)}
                      disabled={selected.size === 0}
                      className={cn(
                        'rounded-[5px] border px-2 py-1 text-[12px]',
                        selected.size === 0
                          ? 'cursor-not-allowed border-[var(--border)] text-[var(--text-subtle)]'
                          : 'border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
                      )}
                    >
                      {lab}
                    </button>
                  ))}
                  <Button onClick={() => setSelected(new Set())} disabled={selected.size === 0}>Bỏ chọn</Button>
                </div>
              </div>

              <div className="min-h-0 flex-1 overflow-auto p-3">
                {gridItems.length === 0 ? (
                  <div className="py-10 text-center text-[12px] text-[var(--text-subtle)]">Không có ảnh.</div>
                ) : (
                  <div className="grid grid-cols-[repeat(auto-fill,minmax(110px,1fr))] gap-2">
                    {gridItems.map((it) => {
                      const isSel = selected.has(it.resultId);
                      const edited = pendingEdits[it.resultId];
                      return (
                        <button
                          key={it.resultId}
                          type="button"
                          onClick={() => toggleSel(it.resultId)}
                          className={cn(
                            'group relative overflow-hidden rounded-[6px] border bg-[var(--surface-2)] text-left',
                            isSel ? 'border-[var(--accent)] ring-2 ring-[var(--accent)]' : 'border-[var(--border-muted)] hover:border-[var(--border)]',
                          )}
                        >
                          {it.cropUri ? (
                            <img src={it.cropUri} alt={`#${it.resultId}`} loading="lazy" className="h-[100px] w-full object-cover" draggable={false} />
                          ) : (
                            <div className="flex h-[100px] items-center justify-center text-[11px] text-[var(--text-subtle)]">no crop</div>
                          )}
                          <div className="flex items-center justify-between px-1.5 py-1 text-[10px]">
                            <span className="text-[var(--text-muted)]">#{it.resultId}</span>
                            <span className={cn('rounded px-1', edited ? 'bg-[var(--accent)] text-white' : 'text-[var(--text)]')}>{it.finalLabel}</span>
                          </div>
                          {isSel && <div className="absolute right-1 top-1 rounded bg-[var(--accent)] px-1 text-[10px] text-white">✓</div>}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex flex-1 items-center justify-center text-[13px] text-[var(--text-muted)]">
              Bấm một lớp ở trên để xem lưới ảnh và soi label.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
