import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, ErrorMessage, Field, SelectControl, TextInput } from '../../components/ui/index.js';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

// "<name>_<YYYYMMDD_HHMMSS>" from an optional English name; '' if no name.
const makeStamped = (name) => {
  const clean = String(name || '').trim().replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
  if (!clean) return '';
  const d = new Date();
  const p2 = (n) => String(n).padStart(2, '0');
  const stamp = `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`;
  return `${clean}_${stamp}`;
};

// Runs step08 (classifier) then step09 (self-training) for the selected run,
// streaming Python output. The "filter name" is used to tag the round.
export default function RunSteps({ dbPath, onChangeDbPath }) {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [resources, setResources] = useState(null);
  const [bridge, setBridge] = useState(null);
  const [filterName, setFilterName] = useState('');
  const [applyPromotions, setApplyPromotions] = useState(true);
  const [confThreshold, setConfThreshold] = useState(0.9);
  const [marginThreshold, setMarginThreshold] = useState(0.1);
  const [log, setLog] = useState('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const logRef = useRef(null);

  useEffect(() => {
    const a = api();
    if (!a) return undefined;
    a.getBridgeInfo().then(setBridge).catch(() => {});
    // subscribe to streamed output
    const off = a.onStepOutput(({ chunk }) => {
      setLog((prev) => prev + chunk);
    });
    return off;
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const loadRuns = useCallback(async () => {
    setError('');
    try {
      const res = await api().listLabelingRuns({ resemiDbPath: dbPath });
      const list = res.runs || [];
      setRuns(list);
      if (list.length && !runId) setRunId(list[0].run_id);
    } catch (e) { setError(String(e?.message || e)); }
  }, [dbPath, runId]);

  const loadResources = useCallback(async () => {
    if (!runId) return;
    try {
      const res = await api().getRunResources({ resemiDbPath: dbPath, runId });
      setResources(res);
    } catch (e) { setError(String(e?.message || e)); }
  }, [dbPath, runId]);

  useEffect(() => { if (runId) loadResources(); }, [runId, loadResources]);

  const runPipeline = useCallback(async () => {
    if (!runId) { setError('Chọn run trước.'); return; }
    setRunning(true);
    setError('');
    setLog('');
    try {
      // step08 — train classifier on audited labels + embeddings
      const step08Flags = {
        '--db': dbPath,
        '--run-id': runId,
        '--model-name': 'facebook/dinov2-small',
        '--view-name': 'tight',
      };
      if (resources?.prototypeVersionId) step08Flags['--prototype-version-id'] = resources.prototypeVersionId;
      if (resources?.coreMiningRunId) step08Flags['--core-mining-run-id'] = resources.coreMiningRunId;

      const r8 = await api().runStep({ step: 'step08', flags: step08Flags, jobId: 'step08' });
      if (r8.code !== 0) { setError('step08 thất bại (xem log).'); setRunning(false); return; }

      // discover the classifier_run_id just created
      const res = await api().getRunResources({ resemiDbPath: dbPath, runId });
      setResources(res);
      const clf = res.classifierRunId;
      if (!clf) { setError('Không tìm thấy classifier_run_id sau step08.'); setRunning(false); return; }

      // step09 — self-training round
      const step09Flags = {
        '--db': dbPath,
        '--run-id': runId,
        '--classifier-run-id': clf,
        '--classifier-confidence-threshold': Number(confThreshold),
        '--classifier-margin-threshold': Number(marginThreshold),
      };
      const stampedName = makeStamped(filterName);
      if (stampedName) step09Flags['--self-training-run-id'] = stampedName;
      if (applyPromotions) step09Flags['--apply-promotions'] = true;

      const r9 = await api().runStep({ step: 'step09', flags: step09Flags, jobId: 'step09' });
      if (r9.code !== 0) { setError('step09 thất bại (xem log).'); setRunning(false); return; }

      await loadResources();
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setRunning(false);
    }
  }, [runId, dbPath, resources, confThreshold, marginThreshold, applyPromotions, filterName, loadResources]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="shrink-0 border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-3">
        <div className="text-[15px] font-semibold text-[var(--text)]">Chạy bước tiếp theo</div>
        <div className="text-[13px] text-[var(--text-muted)]">Train classifier (step08) rồi self-training (step09) trên nhãn đã duyệt.</div>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* controls */}
        <div className="w-[380px] shrink-0 overflow-auto border-r border-[var(--border-muted)] p-5">
          <div className="grid gap-4">
            {error && <ErrorMessage>{error}</ErrorMessage>}
            {bridge && !bridge.pythonExists && (
              <ErrorMessage>Không tìm thấy Python venv ({bridge.python}). Kiểm tra .venv của dự án.</ErrorMessage>
            )}

            <Field label="Resemi DB">
              <div className="flex gap-2">
                <TextInput value={dbPath} onChange={(e) => onChangeDbPath?.(e.currentTarget.value)} className="flex-1" />
                <Button onClick={async () => { const f = await api().browsePath('file'); if (f) onChangeDbPath?.(f); }}>Browse</Button>
              </div>
            </Field>

            <div className="flex items-center gap-2">
              <Button onClick={loadRuns}>Load runs</Button>
              {runs.length > 0 && (
                <SelectControl value={runId} onChange={(e) => setRunId(e.currentTarget.value)} className="flex-1">
                  {runs.map((r) => (
                    <option key={r.run_id} value={r.run_id}>{r.run_id}</option>
                  ))}
                </SelectControl>
              )}
            </div>

            {resources && (
              <div className="grid gap-1 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3 text-[12px]">
                <Row label="review_queue" value={resources.counts?.reviewQueue} />
                <Row label="cleaned_labels" value={resources.counts?.cleaned} />
                <Row label="nhãn người (review_decisions)" value={resources.counts?.reviewDecisions} />
                <Row label="prototype" value={resources.prototypeVersionId || '—'} mono />
                <Row label="core mining" value={resources.coreMiningRunId || '—'} mono />
                <Row label="classifier mới nhất" value={resources.classifierRunId || '—'} mono />
              </div>
            )}

            <Field label="Tên vòng lọc (tuỳ chọn, tiếng Anh)">
              <TextInput value={filterName} onChange={(e) => setFilterName(e.currentTarget.value)} placeholder="round1_clean" />
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Conf ≥">
                <TextInput type="number" step="0.01" min="0" max="1" value={confThreshold} onChange={(e) => setConfThreshold(Number(e.currentTarget.value))} />
              </Field>
              <Field label="Margin ≥">
                <TextInput type="number" step="0.01" min="0" max="1" value={marginThreshold} onChange={(e) => setMarginThreshold(Number(e.currentTarget.value))} />
              </Field>
            </div>

            <label className="flex items-center gap-2 text-[13px] text-[var(--text)]">
              <input type="checkbox" checked={applyPromotions} onChange={(e) => setApplyPromotions(e.currentTarget.checked)} className="accent-[var(--primary)]" />
              Áp dụng promote (xoá box sạch khỏi review_queue)
            </label>

            <Button variant="primary" onClick={runPipeline} disabled={running || !runId}>
              {running ? 'Đang chạy…' : 'Train + Self-train'}
            </Button>
          </div>
        </div>

        {/* log */}
        <div className="flex min-w-0 flex-1 flex-col p-4">
          <div className="mb-2 text-[12px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Output</div>
          <pre
            ref={logRef}
            className="min-h-0 flex-1 overflow-auto rounded-[6px] border border-[var(--border-muted)] bg-[var(--surface-2)] p-3 text-[11px] leading-relaxed text-[var(--text)] whitespace-pre-wrap"
          >
            {log || 'Chưa chạy.'}
          </pre>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, mono }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[var(--text-muted)]">{label}</span>
      <span className={mono ? 'font-mono text-[11px] text-[var(--text)]' : 'text-[var(--text)]'}>{String(value ?? '—')}</span>
    </div>
  );
}
