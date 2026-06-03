import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, ErrorMessage, Field, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';

const api = () => (typeof window !== 'undefined' ? window.electronAPI : null);

// step05 — Prototype bank picker. A gallery (grouped by suggested class) where
// the human multi-selects representative crops per class (and a few rejects),
// then runs step05 via the Python bridge to build the prototype bank.
export default function Prototype({ dbPath, onChangeDbPath, onDataChanged }) {
  const [imageRoot, setImageRoot] = useState('');
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState('');
  const [classLabels, setClassLabels] = useState(['crack', 'mold', 'spall', 'reject']);
  const [activeClass, setActiveClass] = useState('crack');
  const [perBand, setPerBand] = useState(200);
  const [rejectBelow, setRejectBelow] = useState(0.5);
  const [groupMode, setGroupMode] = useState('score'); // 'score' | 'domain'
  const [items, setItems] = useState([]);
  const [modelName, setModelName] = useState('facebook/dinov2-small');
  // picks: resultId -> { label, isReject }
  const [picks, setPicks] = useState({});
  const [existing, setExisting] = useState(null);
  const [screen, setScreen] = useState('setup'); // setup | gallery
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState('');
  const [error, setError] = useState('');
  const logRef = useRef(null);

  useEffect(() => {
    const a = api();
    if (!a) return undefined;
    a.getLabelingDefaults().then((d) => {
      setImageRoot((p) => p || d.imageRootPath || '');
    }).catch(() => {});
    const off = a.onStepOutput(({ chunk }) => setLog((prev) => prev + chunk));
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

  const loadCandidates = useCallback(async () => {
    if (!runId) { setError('Chọn run trước.'); return; }
    setLoading(true);
    setError('');
    try {
      const [res, proto] = await Promise.all([
        api().listPrototypeCandidates({ resemiDbPath: dbPath, runId, imageRootPath: imageRoot, label: 'all', perBand, rejectBelow }),
        api().latestPrototype({ resemiDbPath: dbPath, runId }),
      ]);
      setItems(res.items || []);
      if (Array.isArray(res.labels) && res.labels.length) setClassLabels(res.labels);
      setExisting(proto.prototype || null);
      setScreen('gallery');
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }, [dbPath, runId, imageRoot, perBand, rejectBelow]);

  const togglePick = useCallback((item, isReject) => {
    // a box living in the reject tab is a reject pick even on the main click
    const reject = isReject || item.label === 'reject';
    setPicks((prev) => {
      const next = { ...prev };
      const cur = next[item.resultId];
      if (cur && cur.isReject === reject) {
        delete next[item.resultId]; // toggle off
      } else {
        next[item.resultId] = { label: reject ? 'reject' : item.label, isReject: reject };
      }
      return next;
    });
  }, []);

  const countByClass = useMemo(() => {
    const c = {};
    for (const p of Object.values(picks)) {
      const key = p.isReject ? 'reject' : p.label;
      c[key] = (c[key] || 0) + 1;
    }
    return c;
  }, [picks]);

  const visible = useMemo(
    () => items.filter((it) => it.label === activeClass),
    [items, activeClass],
  );

  // Group the visible candidates either by reliability-score band (≥90%, 80–90%,
  // …) or by step04 core cluster (= domain). Each entry: { key, title, items }.
  const groups = useMemo(() => {
    if (groupMode === 'domain') {
      const m = new Map();
      for (const it of visible) {
        const k = it.domainIndex == null ? -1 : it.domainIndex;
        if (!m.has(k)) m.set(k, { key: k, sort: k < 0 ? 999 : k, clusterSize: it.clusterSize || 0, items: [] });
        m.get(k).items.push(it);
      }
      return [...m.values()]
        .sort((a, b) => a.sort - b.sort)
        .map((g) => ({
          key: `dom-${g.key}`,
          title: g.key < 0 ? 'Khác (ngoài cụm)' : `Domain ${g.key + 1}`,
          sub: g.key < 0 ? '' : `cụm ${g.clusterSize} mẫu`,
          items: g.items,
        }));
    }
    // score bands: 90–100, 80–90, …, <50 lumped at 40
    const m = new Map();
    for (const it of visible) {
      const pct = Math.round((it.reliabilityScore || 0) * 100);
      const band = pct >= 90 ? 90 : pct >= 50 ? Math.floor(pct / 10) * 10 : 40;
      if (!m.has(band)) m.set(band, []);
      m.get(band).push(it);
    }
    return [...m.entries()]
      .sort((a, b) => b[0] - a[0])
      .map(([band, list]) => ({
        key: `score-${band}`,
        title: band === 90 ? '≥ 90%' : band === 40 ? '< 50%' : `${band}–${band + 10}%`,
        sub: '',
        items: list,
      }));
  }, [visible, groupMode]);

  const runStep05 = useCallback(async () => {
    const entries = Object.entries(picks);
    if (!entries.length) { setError('Chưa chọn prototype nào.'); return; }
    const protoArg = entries.filter(([, p]) => !p.isReject).map(([id, p]) => `${id}:${p.label}`).join(',');
    const rejectArg = entries.filter(([, p]) => p.isReject).map(([id]) => `${id}:reject`).join(',');
    if (!protoArg) { setError('Cần ít nhất một prototype không phải reject (step05 cần ≥1 lớp).'); return; }

    setRunning(true);
    setError('');
    setLog('');
    try {
      const flags = {
        '--db': dbPath,
        '--run-id': runId,
        '--model-name': modelName,
        '--view-name': 'tight',
        '--prototype': protoArg,
      };
      if (rejectArg) flags['--reject'] = rejectArg;
      const res = await api().runStep({ step: 'step05', flags, jobId: 'step05' });
      if (res.code !== 0) { setError('step05 thất bại (xem log).'); return; }
      const proto = await api().latestPrototype({ resemiDbPath: dbPath, runId });
      setExisting(proto.prototype || null);
      onDataChanged?.();
      alert('Đã tạo prototype bank. Giờ chạy lại step06→07 ở tab Chạy bước để áp dụng.');
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setRunning(false);
    }
  }, [picks, dbPath, runId, modelName, onDataChanged]);

  if (screen === 'setup') {
    return (
      <div className="rv-enter h-full overflow-auto bg-[var(--bg)] p-8">
        <div className="mx-auto grid w-full max-w-[760px] gap-5">
          <div>
            <div className="text-[15px] font-semibold text-[var(--text)]">Prototype (step05) — chọn mẫu chuẩn</div>
            <div className="text-[13px] text-[var(--text-muted)]">Lướt gallery, chọn ~12-50 crop điển hình mỗi lớp (+ vài mẫu reject) làm chuẩn vàng.</div>
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
            <div className="grid grid-cols-3 gap-3">
              <Field label="DINOv2 model (phải khớp step03)">
                <SelectControl value={modelName} onChange={(e) => setModelName(e.currentTarget.value)}>
                  <option value="facebook/dinov2-small">dinov2-small</option>
                  <option value="facebook/dinov2-large">dinov2-large</option>
                  <option value="facebook/dinov2-giant">dinov2-giant</option>
                </SelectControl>
              </Field>
              <Field label="Số box / dải điểm">
                <TextInput type="number" min={10} max={2000} value={perBand} onChange={(e) => setPerBand(Number(e.currentTarget.value))} />
              </Field>
              <Field label="Reject nếu điểm < (0–1)">
                <TextInput type="number" min={0} max={1} step={0.05} value={rejectBelow} onChange={(e) => setRejectBelow(Number(e.currentTarget.value))} />
              </Field>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={loadRuns}>Load runs</Button>
              {runs.length > 0 && (
                <SelectControl value={runId} onChange={(e) => setRunId(e.currentTarget.value)} className="flex-1">
                  {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}
                </SelectControl>
              )}
            </div>
            <div>
              <Button onClick={loadCandidates} disabled={!runId || loading}>{loading ? 'Loading…' : 'Mở gallery'}</Button>
            </div>
          </section>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg)]">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
        <Button onClick={() => setScreen('setup')}>← Setup</Button>
        <div className="flex items-center gap-1">
          {classLabels.map((lab) => (
            <button
              key={lab}
              type="button"
              onClick={() => setActiveClass(lab)}
              className={cn(
                'rounded-[5px] border px-2 py-1 text-[12px]',
                activeClass === lab
                  ? 'border-[var(--accent)] bg-[var(--active)] text-[var(--text)]'
                  : 'border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--hover)]',
              )}
            >
              {lab} <span className="text-[var(--accent)]">{countByClass[lab] || 0}</span>
            </button>
          ))}
        </div>
        <span className="text-[var(--text-subtle)]">{visible.length} ứng viên {activeClass}</span>
        <div className="ml-2 flex items-center gap-1">
          <span className="text-[var(--text-subtle)]">Nhóm:</span>
          {[['score', 'Điểm'], ['domain', 'Domain']].map(([mode, lab]) => (
            <button
              key={mode}
              type="button"
              onClick={() => setGroupMode(mode)}
              className={cn(
                'rounded-[5px] border px-2 py-1 text-[12px]',
                groupMode === mode
                  ? 'border-[var(--accent)] bg-[var(--active)] text-[var(--text)]'
                  : 'border-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--hover)]',
              )}
            >
              {lab}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          {existing && <span className="text-[var(--text-muted)]">bank hiện tại: {existing.item_count} mẫu</span>}
          <Button variant="primary" onClick={runStep05} disabled={running || Object.keys(picks).length === 0}>
            {running ? 'Đang chạy step05…' : `Tạo bank (${Object.keys(picks).length})`}
          </Button>
        </div>
      </div>

      {error && <div className="px-4 pt-2"><ErrorMessage>{error}</ErrorMessage></div>}

      <div className="min-h-0 flex-1 overflow-auto p-3">
        {groups.length === 0 && (
          <div className="py-10 text-center text-[12px] text-[var(--text-subtle)]">Không có ứng viên cho {activeClass}.</div>
        )}
        {groups.map((d) => (
          <div key={d.key} className="mb-4">
            <div className="mb-1.5 flex items-center gap-2 text-[12px]">
              <span className="font-semibold text-[var(--text)]">{d.title}</span>
              {d.sub && <span className="text-[var(--text-muted)]">· {d.sub}</span>}
              <span className="text-[var(--text-subtle)]">· {d.items.length} ứng viên</span>
              <div className="h-px flex-1 bg-[var(--border-muted)]" />
            </div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2">
              {d.items.map((it) => {
                const pick = picks[it.resultId];
                return (
                  <div
                    key={it.resultId}
                    className={cn(
                      'group relative overflow-hidden rounded-[6px] border bg-[var(--surface-2)]',
                      pick?.isReject ? 'border-[var(--danger,#ef4444)]'
                        : pick ? 'border-[var(--accent)]'
                          : 'border-[var(--border-muted)]',
                    )}
                  >
                    <button type="button" onClick={() => togglePick(it, false)} className="block w-full">
                      {it.cropUri ? (
                        <img src={it.cropUri} alt={`#${it.resultId}`} loading="lazy" className="h-[110px] w-full object-cover" draggable={false} />
                      ) : (
                        <div className="flex h-[110px] items-center justify-center text-[11px] text-[var(--text-subtle)]">no crop</div>
                      )}
                    </button>
                    <div className="flex items-center justify-between px-1.5 py-1 text-[10px]">
                      <span className="text-[var(--text-muted)]">#{it.resultId} · {it.reliabilityScore.toFixed(2)}</span>
                      <button
                        type="button"
                        onClick={() => togglePick(it, true)}
                        className={cn('rounded px-1', pick?.isReject ? 'bg-[var(--danger,#ef4444)] text-white' : 'text-[var(--text-subtle)] hover:text-[var(--danger,#ef4444)]')}
                        title="đánh dấu reject"
                      >
                        ✕
                      </button>
                    </div>
                    {pick && !pick.isReject && (
                      <div className="absolute left-1 top-1 rounded bg-[var(--accent)] px-1 text-[10px] text-white">✓ {pick.label}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {log && (
        <pre ref={logRef} className="max-h-[140px] shrink-0 overflow-auto border-t border-[var(--border-muted)] bg-[var(--surface-2)] p-2 text-[11px] text-[var(--text)] whitespace-pre-wrap">
          {log}
        </pre>
      )}
    </div>
  );
}
