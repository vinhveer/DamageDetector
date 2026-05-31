import { useCallback, useEffect, useState } from 'react';
import { Button, ErrorMessage, Field, SelectControl, TextInput } from '../../components/ui/index.js';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

// R4: export cleaned_labels to a YOLO/COCO dataset.
export default function Export({ dbPath, onChangeDbPath }) {
  const [imageRoot, setImageRoot] = useState('');
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [format, setFormat] = useState('both');
  const [outputDir, setOutputDir] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

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

  const run = useCallback(async () => {
    if (!runId) { setError('Chọn run trước.'); return; }
    if (!outputDir) { setError('Chọn thư mục đích.'); return; }
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const res = await api().exportDataset({
        resemiDbPath: dbPath, runId, imageRootPath: imageRoot, format, outputDir,
      });
      if (res.error) { setError(res.error); return; }
      setResult(res);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  }, [dbPath, runId, imageRoot, format, outputDir]);

  return (
    <div className="rv-enter h-full overflow-auto bg-[var(--bg)] p-8">
      <div className="mx-auto grid w-full max-w-[760px] gap-5">
        <div>
          <div className="text-[15px] font-semibold text-[var(--text)]">Export dataset</div>
          <div className="text-[13px] text-[var(--text-muted)]">Xuất cleaned_labels ra YOLO / COCO (đã áp dụng export_label, bỏ reject).</div>
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
                {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.cleaned_count} cleaned</option>)}
              </SelectControl>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Format">
              <SelectControl value={format} onChange={(e) => setFormat(e.currentTarget.value)}>
                <option value="both">both (YOLO + COCO)</option>
                <option value="yolo">yolo</option>
                <option value="coco">coco</option>
              </SelectControl>
            </Field>
            <Field label="Thư mục đích">
              <div className="flex gap-2">
                <TextInput value={outputDir} onChange={(e) => setOutputDir(e.currentTarget.value)} className="flex-1" />
                <Button onClick={async () => { const f = await api().browsePath('directory'); if (f) setOutputDir(f); }}>Browse</Button>
              </div>
            </Field>
          </div>
          <div>
            <Button variant="primary" onClick={run} disabled={busy || !runId}>{busy ? 'Đang xuất…' : 'Xuất'}</Button>
          </div>
        </section>

        {result && (
          <section className="grid gap-1 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-5 text-[13px]">
            <div className="text-[12px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Kết quả</div>
            <Row label="ảnh đã ghi" value={result.images_written} />
            <Row label="box đã ghi" value={result.boxes_written} />
            <Row label="box bị loại (reject)" value={result.boxes_rejected} />
            <Row label="ảnh bị bỏ qua" value={result.images_skipped} />
            <Row label="categories" value={(result.categories || []).join(', ')} />
            <Row label="output" value={result.output_dir} mono />
          </section>
        )}
      </div>
    </div>
  );
}

function Row({ label, value, mono }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[var(--text-muted)]">{label}</span>
      <span className={mono ? 'break-all font-mono text-[11px] text-[var(--text)]' : 'text-[var(--text)]'}>{String(value ?? '—')}</span>
    </div>
  );
}
