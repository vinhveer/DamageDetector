import { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, EmptyState, ErrorMessage, Field, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';
import BoxImage from './BoxImage.jsx';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

// R2 + R3: view and edit cleaned_labels (machine-promoted final labels),
// committing fixes as a new "corrections" review session.
export default function CleanedLabels({ dbPath, onChangeDbPath, dataVersion }) {
  const [imageRoot, setImageRoot] = useState('');
  const [labels, setLabels] = useState(['crack', 'mold', 'spall', 'stain', 'reject']);
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [index, setIndex] = useState(0);
  const [labelFilter, setLabelFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  // pendingEdits keyed by resultId -> { action, previousLabel, newLabel }
  const [pendingEdits, setPendingEdits] = useState({});
  const [sessionName, setSessionName] = useState('');
  const [screen, setScreen] = useState('setup');
  const [loading, setLoading] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const a = api();
    if (!a) return;
    a.getLabelingDefaults()
      .then((d) => {
        setImageRoot((prev) => prev || d.imageRootPath || '');
        if (Array.isArray(d.labels) && d.labels.length) setLabels(d.labels);
      })
      .catch(() => {});
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

  const load = useCallback(async () => {
    if (!runId) { setError('Chọn một run trước.'); return; }
    setLoading(true);
    setError('');
    try {
      const res = await api().listCleaned({
        resemiDbPath: dbPath,
        runId,
        imageRootPath: imageRoot,
        finalLabel: labelFilter,
        decisionType: typeFilter,
      });
      setItems(res.items || []);
      setTotal(res.total ?? (res.items || []).length);
      setIndex(0);
      setScreen('review');
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, [dbPath, runId, imageRoot, labelFilter, typeFilter]);

  // reload when switching back into the tab after a step run
  useEffect(() => {
    if (screen === 'review' && runId) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataVersion]);

  const current = items[index] || null;
  const pendingCount = useMemo(() => Object.keys(pendingEdits).length, [pendingEdits]);

  const goNext = useCallback(() => setIndex((i) => Math.min(items.length - 1, i + 1)), [items.length]);
  const goPrev = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  const applyEdit = useCallback(async (action, newLabel) => {
    if (!current) return;
    const prev = current.finalLabel;
    const effectiveLabel = action === 'manual_reject' ? 'reject' : (action === 'manual_accept' ? prev : newLabel);
    try {
      const res = await api().updateCleanedLabel({
        resemiDbPath: dbPath, runId, resultId: current.resultId, newLabel: effectiveLabel,
      });
      if (res.error) { setError(res.error); return; }
      // update the in-memory item with the persisted values
      setItems((arr) => arr.map((it) => (it.resultId === current.resultId
        ? { ...it, finalLabel: res.finalLabel ?? effectiveLabel, exportLabel: res.exportLabel ?? it.exportLabel }
        : it)));
      setPendingEdits((p) => ({ ...p, [current.resultId]: { action, previousLabel: prev, newLabel: effectiveLabel } }));
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [current, dbPath, runId]);

  const commit = useCallback(async () => {
    const edits = Object.entries(pendingEdits).map(([resultId, d]) => ({
      resultId: Number(resultId), action: d.action, previousLabel: d.previousLabel, newLabel: d.newLabel,
    }));
    if (!edits.length) { setError('Chưa có chỉnh sửa nào'); return; }
    setCommitting(true);
    setError('');
    try {
      const res = await api().commitCorrections({ resemiDbPath: dbPath, runId, sessionName, edits });
      if (res.error) { setError(res.error); return; }
      alert(`Đã ghi ${res.decisionCount} chỉnh sửa vào session ${res.reviewSessionId}`);
      setPendingEdits({});
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setCommitting(false);
    }
  }, [pendingEdits, dbPath, runId, sessionName]);

  if (screen === 'setup') {
    return (
      <div className="rv-enter h-full overflow-auto bg-[var(--bg)] p-8">
        <div className="mx-auto grid w-full max-w-[760px] gap-5">
          <div>
            <div className="text-[15px] font-semibold text-[var(--text)]">Cleaned labels — xem & sửa</div>
            <div className="text-[13px] text-[var(--text-muted)]">Nhãn máy đã tự chốt (step07 auto-accept + step09 promote). Sửa cái sai rồi commit thành phiên corrections.</div>
          </div>
          {error && <ErrorMessage>{error}</ErrorMessage>}
          <section className="grid gap-4 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-5">
            <Field label="Resemi DB (resemi.sqlite3)">
              <div className="flex gap-2">
                <TextInput value={dbPath} onChange={(e) => onChangeDbPath?.(e.currentTarget.value)} className="flex-1" />
                <Button onClick={async () => { const f = await api().browsePath('file'); if (f) onChangeDbPath?.(f); }}>Browse</Button>
              </div>
            </Field>
            <Field label="Image root">
              <div className="flex gap-2">
                <TextInput value={imageRoot} onChange={(e) => setImageRoot(e.currentTarget.value)} className="flex-1" />
                <Button onClick={async () => { const f = await api().browsePath('directory'); if (f) setImageRoot(f); }}>Browse</Button>
              </div>
            </Field>
            <div className="flex items-center gap-2">
              <Button onClick={loadRuns}>Load runs</Button>
              {runs.length > 0 && (
                <SelectControl value={runId} onChange={(e) => setRunId(e.currentTarget.value)} className="flex-1">
                  {runs.map((r) => (
                    <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.cleaned_count} cleaned</option>
                  ))}
                </SelectControl>
              )}
            </div>
            <div>
              <Button onClick={load} disabled={!runId || loading}>{loading ? 'Loading…' : 'Mở cleaned labels'}</Button>
            </div>
          </section>
        </div>
      </div>
    );
  }

  if (!current) {
    return (
      <div className="flex h-full flex-col bg-[var(--bg)]">
        <TopFilters {...{ labels, labelFilter, setLabelFilter, typeFilter, setTypeFilter, load, setScreen, total, pendingCount }} />
        <div className="flex flex-1 items-center justify-center">
          <EmptyState title="Không có bản ghi" message="cleaned_labels rỗng cho bộ lọc này." />
        </div>
      </div>
    );
  }

  const edited = pendingEdits[current.resultId];

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
        <Button onClick={() => setScreen('setup')}>← Setup</Button>
        <span className="text-[var(--text-muted)]">{index + 1} / {items.length}</span>
        <span className="text-[var(--text-subtle)]">(tổng {total} cleaned)</span>
        <span className="text-[var(--text)]">đã sửa: {pendingCount}</span>
        <div className="ml-auto flex items-center gap-2">
          <SelectControl value={labelFilter} onChange={(e) => setLabelFilter(e.currentTarget.value)}>
            <option value="all">nhãn: tất cả</option>
            {labels.map((l) => <option key={l} value={l}>{l}</option>)}
          </SelectControl>
          <SelectControl value={typeFilter} onChange={(e) => setTypeFilter(e.currentTarget.value)}>
            <option value="all">loại: tất cả</option>
            <option value="auto_accept">auto_accept</option>
            <option value="auto_accept_low_priority">auto_accept_low_priority</option>
            <option value="self_training_promote">self_training_promote</option>
          </SelectControl>
          <Button onClick={load}>Áp dụng lọc</Button>
          <TextInput value={sessionName} onChange={(e) => setSessionName(e.currentTarget.value)} placeholder="tên phiên" className="w-[150px]" />
          <Button onClick={commit} disabled={committing || pendingCount === 0}>{committing ? 'Committing…' : `Commit (${pendingCount})`}</Button>
        </div>
      </div>

      {error && <div className="px-4 pt-2"><ErrorMessage>{error}</ErrorMessage></div>}

      <div className="flex min-h-0 flex-1">
        <div className="min-w-0 flex-1 p-4">
          <BoxImage imageUri={current.imageUri} cropUri={current.cropUri} box={current.box} />
        </div>

        <div className="flex w-[300px] shrink-0 flex-col gap-3 overflow-auto border-l border-[var(--border-muted)] bg-[var(--surface)] p-4 text-[13px]">
          <div className="grid gap-1">
            <div className="text-[var(--text-muted)]">result_id <span className="text-[var(--text)]">{current.resultId}</span></div>
            <div className="text-[var(--text-muted)]">final_label <span className="text-[var(--text)]">{current.finalLabel}</span></div>
            <div className="text-[var(--text-muted)]">export_label <span className="text-[var(--text)]">{current.exportLabel}</span></div>
            <div className="text-[var(--text-muted)]">decision <span className="text-[var(--text)]">{current.decisionType}</span></div>
            <div className="text-[var(--text-muted)]">reliability <span className="text-[var(--text)]">{current.reliabilityScore.toFixed(2)}</span></div>
            {current.selfTrainingRunId && (
              <div className="text-[var(--text-muted)]">self-train <span className="break-all text-[11px] text-[var(--text)]">{current.selfTrainingRunId}</span></div>
            )}
          </div>

          <div className="grid gap-2 border-t border-[var(--border-muted)] pt-3">
            <Button onClick={() => applyEdit('manual_accept')}>✓ Giữ nguyên ({current.finalLabel})</Button>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Sửa nhãn</div>
            <div className="grid grid-cols-2 gap-2">
              {labels.map((lab) => {
                const active = edited?.newLabel === lab;
                const onClick = lab === 'reject'
                  ? () => applyEdit('manual_reject')
                  : () => applyEdit('manual_relabel', lab);
                return (
                  <button
                    key={lab}
                    type="button"
                    onClick={onClick}
                    className={cn(
                      'rounded-[5px] border px-2 py-1.5 text-[12px] font-medium',
                      active
                        ? 'border-[var(--accent)] bg-[var(--active)] text-[var(--text)]'
                        : 'border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
                    )}
                  >
                    {lab}
                  </button>
                );
              })}
            </div>
          </div>

          {edited && (
            <div className="text-[12px] text-[var(--accent)]">
              đã sửa: {edited.action === 'manual_reject' ? 'reject' : edited.action === 'manual_accept' ? 'giữ nguyên' : 'relabel'} → {edited.newLabel}
            </div>
          )}

          <div className="mt-auto flex gap-2 border-t border-[var(--border-muted)] pt-3">
            <Button onClick={goPrev} disabled={index === 0}>← Lùi</Button>
            <Button onClick={goNext} disabled={index >= items.length - 1}>Tiếp →</Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function TopFilters({ labels, labelFilter, setLabelFilter, typeFilter, setTypeFilter, load, setScreen, total, pendingCount }) {
  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
      <Button onClick={() => setScreen('setup')}>← Setup</Button>
      <span className="text-[var(--text-subtle)]">tổng {total} cleaned</span>
      <span className="text-[var(--text)]">đã sửa: {pendingCount}</span>
      <div className="ml-auto flex items-center gap-2">
        <SelectControl value={labelFilter} onChange={(e) => setLabelFilter(e.currentTarget.value)}>
          <option value="all">nhãn: tất cả</option>
          {labels.map((l) => <option key={l} value={l}>{l}</option>)}
        </SelectControl>
        <SelectControl value={typeFilter} onChange={(e) => setTypeFilter(e.currentTarget.value)}>
          <option value="all">loại: tất cả</option>
          <option value="auto_accept">auto_accept</option>
          <option value="auto_accept_low_priority">auto_accept_low_priority</option>
          <option value="self_training_promote">self_training_promote</option>
        </SelectControl>
        <Button onClick={load}>Áp dụng lọc</Button>
      </div>
    </div>
  );
}
