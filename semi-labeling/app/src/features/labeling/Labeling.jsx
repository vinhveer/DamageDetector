import { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, EmptyState, ErrorMessage, Field, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';
import BoxImage from './BoxImage.jsx';
import { reasonLabelVi } from './reasonLabels.js';

const ACTIONS = {
  ACCEPT: 'manual_accept',
  RELABEL: 'manual_relabel',
  REJECT: 'manual_reject',
};

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

export default function Labeling() {
  const [paths, setPaths] = useState({ resemiDbPath: '', imageRootPath: '' });
  const [labels, setLabels] = useState(['crack', 'mold', 'spall', 'stain', 'reject']);
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [items, setItems] = useState([]);
  const [index, setIndex] = useState(0);
  const [samplePercent, setSamplePercent] = useState(10);
  const [queueTotal, setQueueTotal] = useState(0);
  // decisions keyed by resultId -> { action, newLabel }
  const [decisions, setDecisions] = useState({});
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [screen, setScreen] = useState('setup'); // setup | review
  const [committing, setCommitting] = useState(false);
  const [sessionName, setSessionName] = useState('');

  useEffect(() => {
    const a = api();
    if (!a) return;
    a.getLabelingDefaults()
      .then((d) => {
        setPaths({ resemiDbPath: d.resemiDbPath || '', imageRootPath: d.imageRootPath || '' });
        if (Array.isArray(d.labels) && d.labels.length) setLabels(d.labels);
      })
      .catch((e) => setError(String(e?.message || e)));
  }, []);

  const loadRuns = useCallback(async () => {
    setError('');
    try {
      const res = await api().listLabelingRuns({ resemiDbPath: paths.resemiDbPath });
      const list = res.runs || [];
      setRuns(list);
      if (list.length && !runId) setRunId(list[0].run_id);
    } catch (e) {
      setError(String(e?.message || e));
    }
  }, [paths.resemiDbPath, runId]);

  const startReview = useCallback(async () => {
    if (!runId) { setError('Chọn một run trước.'); return; }
    setLoading(true);
    setError('');
    try {
      const pct = Number(samplePercent);
      const ratio = Number.isFinite(pct) && pct > 0 && pct < 100 ? pct / 100 : 0;
      const res = await api().listLabelingQueue({
        resemiDbPath: paths.resemiDbPath,
        runId,
        imageRootPath: paths.imageRootPath,
        queueType: 'all',
        sampleRatio: ratio,
      });
      setItems(res.items || []);
      setQueueTotal(res.queueTotal ?? (res.items || []).length);
      setIndex(0);
      // seed decisions from any previously committed rows
      const seeded = {};
      for (const it of res.items || []) {
        if (it.decidedAction) seeded[it.resultId] = { action: it.decidedAction, newLabel: it.decidedLabel || it.suggestedLabel };
      }
      setDecisions(seeded);
      setScreen('review');
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, [paths, runId, samplePercent]);

  const current = items[index] || null;
  const decidedCount = useMemo(() => Object.keys(decisions).length, [decisions]);

  const setDecision = useCallback((resultId, action, newLabel) => {
    setDecisions((prev) => ({ ...prev, [resultId]: { action, newLabel: newLabel || '' } }));
  }, []);

  const goNext = useCallback(() => setIndex((i) => Math.min(items.length - 1, i + 1)), [items.length]);
  const goPrev = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  const acceptSuggested = useCallback(() => {
    if (!current) return;
    setDecision(current.resultId, ACTIONS.ACCEPT, current.suggestedLabel || current.initialLabel);
    goNext();
  }, [current, setDecision, goNext]);

  const relabel = useCallback((label) => {
    if (!current) return;
    const action = label === 'reject' ? ACTIONS.REJECT : ACTIONS.RELABEL;
    setDecision(current.resultId, action, label);
    goNext();
  }, [current, setDecision, goNext]);

  // Keyboard: 1-5 = labels, Enter = accept suggested, Space = next, Backspace = prev
  useEffect(() => {
    if (screen !== 'review') return undefined;
    const onKey = (e) => {
      if (e.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (e.key === 'Enter') { e.preventDefault(); acceptSuggested(); }
      else if (e.key === ' ') { e.preventDefault(); goNext(); }
      else if (e.key === 'Backspace') { e.preventDefault(); goPrev(); }
      else if (/^[1-9]$/.test(e.key)) {
        const lab = labels[Number(e.key) - 1];
        if (lab) { e.preventDefault(); relabel(lab); }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [screen, labels, acceptSuggested, goNext, goPrev, relabel]);

  const commit = useCallback(async () => {
    const list = Object.entries(decisions).map(([resultId, d]) => {
      const it = items.find((x) => String(x.resultId) === String(resultId));
      return {
        resultId: Number(resultId),
        action: d.action,
        previousLabel: it ? it.initialLabel : '',
        newLabel: d.newLabel,
      };
    });
    if (!list.length) { setError('Chưa gán nhãn nào.'); return; }
    setCommitting(true);
    setError('');
    try {
      const res = await api().commitLabeling({ resemiDbPath: paths.resemiDbPath, runId, decisions: list, sessionName });
      if (res.error) { setError(res.error); return; }
      setError('');
      alert(`Đã commit ${res.decisionCount} nhãn vào session ${res.reviewSessionId}`);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setCommitting(false);
    }
  }, [decisions, items, paths.resemiDbPath, runId, sessionName]);

  if (screen === 'setup') {
    return (
      <div className="rv-enter h-full overflow-auto bg-[var(--bg)] p-8">
        <div className="mx-auto grid w-full max-w-[760px] gap-5">
          <div>
            <div className="text-[15px] font-semibold text-[var(--text)]">Labeling — review queue</div>
            <div className="text-[13px] text-[var(--text-muted)]">Gán nhãn tay cho các box trong review_queue (step07).</div>
          </div>
          {error && <ErrorMessage>{error}</ErrorMessage>}
          <section className="grid gap-4 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-5">
            <Field label="Resemi DB (resemi.sqlite3)">
              <div className="flex gap-2">
                <TextInput value={paths.resemiDbPath} onChange={(e) => setPaths((p) => ({ ...p, resemiDbPath: e.currentTarget.value }))} className="flex-1" />
                <Button onClick={async () => { const f = await api().browsePath('file'); if (f) setPaths((p) => ({ ...p, resemiDbPath: f })); }}>Browse</Button>
              </div>
            </Field>
            <Field label="Image root">
              <div className="flex gap-2">
                <TextInput value={paths.imageRootPath} onChange={(e) => setPaths((p) => ({ ...p, imageRootPath: e.currentTarget.value }))} className="flex-1" />
                <Button onClick={async () => { const f = await api().browsePath('directory'); if (f) setPaths((p) => ({ ...p, imageRootPath: f })); }}>Browse</Button>
              </div>
            </Field>
            <div className="flex items-center gap-2">
              <Button onClick={loadRuns}>Load runs</Button>
              {runs.length > 0 && (
                <SelectControl value={runId} onChange={(e) => setRunId(e.currentTarget.value)} className="flex-1">
                  {runs.map((r) => (
                    <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.queue_count} review / {r.cleaned_count} cleaned</option>
                  ))}
                </SelectControl>
              )}
            </div>
            <Field label="Tỉ lệ mẫu cần label (%)">
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={samplePercent}
                  onChange={(e) => setSamplePercent(Number(e.currentTarget.value))}
                  className="flex-1 accent-[var(--primary)]"
                />
                <div className="flex items-center gap-1">
                  <TextInput
                    type="number"
                    min={1}
                    max={100}
                    value={samplePercent}
                    onChange={(e) => setSamplePercent(Number(e.currentTarget.value))}
                    className="w-[64px] text-right"
                  />
                  <span className="text-[13px] text-[var(--text-muted)]">%</span>
                </div>
              </div>
              <span className="text-[11px] text-[var(--text-subtle)]">
                Chọn mẫu đa dạng nhất theo embedding (chia đều mỗi lớp). 100% = label toàn bộ queue.
              </span>
            </Field>
            <div>
              <Button onClick={startReview} disabled={!runId || loading}>{loading ? 'Loading…' : 'Bắt đầu review'}</Button>
            </div>
          </section>
        </div>
      </div>
    );
  }

  // review screen
  if (!current) {
    return (
      <div className="flex h-full items-center justify-center bg-[var(--bg)]">
        <EmptyState title="Hết hàng review" message="review_queue rỗng cho run này." />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      {/* top bar */}
      <div className="flex shrink-0 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
        <Button onClick={() => setScreen('setup')}>← Setup</Button>
        <span className="text-[var(--text-muted)]">{index + 1} / {items.length}</span>
        {queueTotal > items.length && (
          <span className="text-[var(--text-subtle)]">(mẫu {items.length} / {queueTotal} queue)</span>
        )}
        <span className="text-[var(--text)]">đã gán: {decidedCount}</span>
        <span className="ml-auto text-[var(--text-muted)]">phím: 1-{labels.length} gán nhãn · Enter nhận gợi ý · Space tiếp · ⌫ lùi</span>
        <TextInput
          value={sessionName}
          onChange={(e) => setSessionName(e.currentTarget.value)}
          placeholder="tên phiên (tuỳ chọn)"
          className="w-[180px]"
        />
        <Button onClick={commit} disabled={committing || decidedCount === 0}>{committing ? 'Committing…' : `Commit (${decidedCount})`}</Button>
      </div>

      {error && <div className="px-4 pt-2"><ErrorMessage>{error}</ErrorMessage></div>}

      <div className="flex min-h-0 flex-1">
        {/* image with box */}
        <div className="min-w-0 flex-1 p-4">
          <BoxImage imageUri={current.imageUri} cropUri={current.cropUri} box={current.box} />
        </div>

        {/* side panel */}
        <div className="flex w-[300px] shrink-0 flex-col gap-3 overflow-auto border-l border-[var(--border-muted)] bg-[var(--surface)] p-4 text-[13px]">
          <div className="grid gap-1">
            <div className="text-[var(--text-muted)]">result_id <span className="text-[var(--text)]">{current.resultId}</span></div>
            <div className="text-[var(--text-muted)]">queue <span className="text-[var(--text)]">{current.queueType}</span></div>
            <div className="text-[var(--text-muted)]">reliability <span className="text-[var(--text)]">{current.reliabilityScore.toFixed(2)}</span></div>
            <div className="text-[var(--text-muted)]">gợi ý (step07) <span className="text-[var(--text)]">{current.suggestedLabel || current.initialLabel}</span></div>
            <div className="text-[var(--text-muted)]">nhãn gốc <span className="text-[var(--text)]">{current.initialLabel}</span></div>
          </div>

          {/* R1: classifier prediction + defer reasons */}
          <div className="grid gap-1 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] p-2.5">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Máy đoán</div>
            {current.prediction ? (
              <>
                <div className="text-[var(--text)]">
                  {current.prediction.predictedLabel}
                  <span className="ml-1 text-[var(--text-muted)]">
                    {(current.prediction.predictedProbability * 100).toFixed(1)}% · margin {current.prediction.margin.toFixed(2)}
                  </span>
                </div>
                {current.prediction.secondLabel && (
                  <div className="text-[12px] text-[var(--text-muted)]">
                    kế tiếp: {current.prediction.secondLabel}
                    {current.prediction.secondProbability != null && ` (${(current.prediction.secondProbability * 100).toFixed(1)}%)`}
                  </div>
                )}
                {current.prediction.disagreesWithPolicy && (
                  <div className="text-[12px] text-[var(--accent,#f43f5e)]">
                    ⚠ máy không đồng ý policy{current.prediction.policyLabel ? ` (policy: ${current.prediction.policyLabel})` : ''}
                  </div>
                )}
              </>
            ) : (
              <div className="text-[12px] text-[var(--text-subtle)]">chưa có dự đoán — dùng gợi ý step07</div>
            )}
            {current.deferReasons.length > 0 && (
              <div className="mt-1 grid gap-1">
                <div className="text-[11px] text-[var(--text-muted)]">lý do defer:</div>
                <div className="flex flex-wrap gap-1">
                  {current.deferReasons.map((r) => (
                    <span key={r} className="rounded-[4px] bg-[var(--hover)] px-1.5 py-0.5 text-[11px] text-[var(--text-muted)]">{reasonLabelVi(r)}</span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {current.reasons.length > 0 && (
            <div className="grid gap-1">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">reasons</div>
              <div className="flex flex-wrap gap-1">
                {current.reasons.map((r) => (
                  <span key={r} className="rounded-[4px] bg-[var(--hover)] px-1.5 py-0.5 text-[11px] text-[var(--text-muted)]">{r}</span>
                ))}
              </div>
            </div>
          )}

          <div className="grid gap-2 border-t border-[var(--border-muted)] pt-3">
            <Button onClick={acceptSuggested}>✓ Nhận gợi ý ({current.suggestedLabel || current.initialLabel}) · Enter</Button>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Gán nhãn khác</div>
            <div className="grid grid-cols-2 gap-2">
              {labels.map((lab, i) => {
                const active = decisions[current.resultId]?.newLabel === lab;
                return (
                  <button
                    key={lab}
                    type="button"
                    onClick={() => relabel(lab)}
                    className={cn(
                      'rounded-[5px] border px-2 py-1.5 text-[12px] font-medium',
                      active
                        ? 'border-[var(--accent)] bg-[var(--active)] text-[var(--text)]'
                        : 'border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
                    )}
                  >
                    {i + 1}. {lab}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="mt-auto flex gap-2 border-t border-[var(--border-muted)] pt-3">
            <Button onClick={goPrev} disabled={index === 0}>← Lùi</Button>
            <Button onClick={goNext} disabled={index >= items.length - 1}>Tiếp →</Button>
          </div>
          {decisions[current.resultId] && (
            <div className="text-[12px] text-[var(--accent)]">
              đã chọn: {decisions[current.resultId].action === 'manual_accept' ? 'accept' : decisions[current.resultId].action === 'manual_reject' ? 'reject' : 'relabel'} → {decisions[current.resultId].newLabel}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
