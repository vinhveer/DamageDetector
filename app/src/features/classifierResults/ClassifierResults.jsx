import { useCallback, useEffect, useMemo, useState } from 'react';
import { IconChartHistogram, IconLayoutGrid, IconRefresh } from '@tabler/icons-react';
import BoxThumbnail from '../clusterLabeling/components/BoxThumbnail.jsx';

const CLASS_COLOR = {
  crack: '#fbbf24',
  mold: '#10b981',
  spall: '#60a5fa',
  reject: '#ef4444',
};

const CONFIDENCE_BUCKETS = [
  { value: 'all', label: 'All confidence' },
  { value: '>0.95', label: '>0.95 (very confident)' },
  { value: '0.80-0.95', label: '0.80 – 0.95' },
  { value: '0.50-0.80', label: '0.50 – 0.80' },
  { value: '<=0.50', label: '≤ 0.50 (low conf)' },
];

const formatPct = (v) => `${(v * 100).toFixed(2)}%`;

const formatSince = (iso) => {
  if (!iso) return '';
  const ts = Date.parse(iso);
  if (!ts) return '';
  const secs = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
};

function MetricsCards({ apply, metrics }) {
  const best = metrics?.best_model || 'mlp';
  const mlpAcc = metrics?.mlp?.test_accuracy ?? null;
  const lrAcc = metrics?.logreg?.test_accuracy ?? null;
  const cv = apply?.cv || null;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Card title="Best model">
        <div className="text-[18px] font-semibold uppercase text-[var(--primary)]">{best}</div>
        <div className="text-[11px] text-[var(--text-muted)]">
          {metrics?.n_total_labeled?.toLocaleString() || '—'} training samples
        </div>
      </Card>
      <Card title="Test accuracy">
        <div className="text-[18px] font-semibold text-[var(--text)]">
          {mlpAcc != null ? formatPct(mlpAcc) : '—'}
        </div>
        <div className="text-[11px] text-[var(--text-muted)]">
          LR baseline: {lrAcc != null ? formatPct(lrAcc) : '—'}
        </div>
      </Card>
      <Card title={cv ? `${cv.folds.length}-fold CV` : 'CV'}>
        <div className="text-[18px] font-semibold text-[var(--text)]">
          {cv ? `${formatPct(cv.mean)} ± ${(cv.std * 100).toFixed(2)}%` : '—'}
        </div>
        <div className="text-[11px] text-[var(--text-muted)]">
          {cv ? `folds: ${cv.folds.map((f) => formatPct(f)).join(', ')}` : 'CV chưa chạy'}
        </div>
      </Card>
      <Card title="Predictions">
        <div className="text-[18px] font-semibold text-[var(--text)]">
          {apply?.n_predictions?.toLocaleString() || '—'}
        </div>
        <div className="text-[11px] text-[var(--text-muted)]">
          Low-conf (&lt; {apply?.low_conf_threshold || 0.5}): {apply?.low_conf_count || 0}
        </div>
      </Card>
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-3">
      <div className="text-[10px] font-medium uppercase tracking-wide text-[var(--text-muted)]">{title}</div>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

function ConfusionMatrix({ classes, matrix, title }) {
  if (!matrix || !classes?.length) return null;
  const totals = matrix.map((row) => row.reduce((sum, v) => sum + v, 0));
  const max = Math.max(...matrix.flat());

  return (
    <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-2 text-[12px] font-semibold uppercase tracking-wide text-[var(--text)]">
        {title}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[11px]">
          <thead>
            <tr>
              <th className="border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1.5 text-left text-[var(--text-muted)]">
                actual ↓ / predicted →
              </th>
              {classes.map((c) => (
                <th
                  key={c}
                  className="border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1.5 font-semibold"
                  style={{ color: CLASS_COLOR[c] || 'var(--text)' }}
                >
                  {c}
                </th>
              ))}
              <th className="border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1.5 text-[var(--text-muted)]">
                recall
              </th>
            </tr>
          </thead>
          <tbody>
            {classes.map((rowLabel, i) => {
              const total = totals[i];
              const recall = total > 0 ? matrix[i][i] / total : 0;
              return (
                <tr key={rowLabel}>
                  <td
                    className="border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1.5 font-semibold"
                    style={{ color: CLASS_COLOR[rowLabel] || 'var(--text)' }}
                  >
                    {rowLabel}
                  </td>
                  {matrix[i].map((value, j) => {
                    const onDiagonal = i === j;
                    const intensity = max > 0 ? value / max : 0;
                    const bg = onDiagonal
                      ? `rgba(16, 185, 129, ${0.15 + intensity * 0.5})`
                      : `rgba(239, 68, 68, ${intensity * 0.4})`;
                    return (
                      <td
                        key={`${rowLabel}-${classes[j]}`}
                        className="border border-[var(--border)] px-2 py-1.5 text-center tabular-nums"
                        style={{ backgroundColor: bg }}
                      >
                        {value}
                      </td>
                    );
                  })}
                  <td className="border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1.5 text-center font-mono text-[var(--text-muted)]">
                    {(recall * 100).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PerClassMetrics({ classes, report, title }) {
  if (!report || !classes?.length) return null;
  return (
    <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-2 text-[12px] font-semibold uppercase tracking-wide text-[var(--text)]">
        {title}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-[var(--border)] text-left text-[var(--text-muted)]">
              <th className="px-2 py-1.5">Class</th>
              <th className="px-2 py-1.5 text-right">Precision</th>
              <th className="px-2 py-1.5 text-right">Recall</th>
              <th className="px-2 py-1.5 text-right">F1</th>
              <th className="px-2 py-1.5 text-right">Support</th>
            </tr>
          </thead>
          <tbody>
            {classes.map((cls) => {
              const row = report[cls] || {};
              return (
                <tr key={cls} className="border-b border-[var(--border-muted)]">
                  <td className="px-2 py-1.5 font-semibold" style={{ color: CLASS_COLOR[cls] || 'var(--text)' }}>
                    {cls}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono">{(row.precision || 0).toFixed(3)}</td>
                  <td className="px-2 py-1.5 text-right font-mono">{(row.recall || 0).toFixed(3)}</td>
                  <td className="px-2 py-1.5 text-right font-mono">{(row['f1-score'] || 0).toFixed(3)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-[var(--text-muted)]">{row.support || 0}</td>
                </tr>
              );
            })}
            {report['macro avg'] && (
              <tr className="text-[var(--text-muted)]">
                <td className="px-2 py-1.5 font-medium">macro avg</td>
                <td className="px-2 py-1.5 text-right font-mono">{(report['macro avg'].precision || 0).toFixed(3)}</td>
                <td className="px-2 py-1.5 text-right font-mono">{(report['macro avg'].recall || 0).toFixed(3)}</td>
                <td className="px-2 py-1.5 text-right font-mono">{(report['macro avg']['f1-score'] || 0).toFixed(3)}</td>
                <td className="px-2 py-1.5 text-right font-mono">—</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PredictionDistribution({ classDist, total }) {
  const entries = Object.entries(classDist || {}).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;
  return (
    <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-3 text-[12px] font-semibold uppercase tracking-wide text-[var(--text)]">
        Predicted class distribution (needs_split)
      </div>
      <div className="space-y-2">
        {entries.map(([cls, count]) => {
          const pct = total > 0 ? (count / total) * 100 : 0;
          const color = CLASS_COLOR[cls] || 'var(--primary)';
          return (
            <div key={cls}>
              <div className="mb-1 flex items-center justify-between text-[11px]">
                <span className="font-semibold" style={{ color }}>{cls}</span>
                <span className="font-mono text-[var(--text-muted)]">
                  {count.toLocaleString()} · {pct.toFixed(1)}%
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--surface-2)]">
                <div
                  className="h-full transition-all"
                  style={{ width: `${pct}%`, backgroundColor: color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PredictionsBrowser({ predictions, classes }) {
  const [classFilter, setClassFilter] = useState('all');
  const [confFilter, setConfFilter] = useState('all');
  const [sortBy, setSortBy] = useState('conf_desc');
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 36;

  useEffect(() => { setPage(0); }, [classFilter, confFilter, sortBy]);

  const filtered = useMemo(() => {
    let list = predictions;
    if (classFilter !== 'all') {
      list = list.filter((p) => p.predicted_class === classFilter);
    }
    if (confFilter !== 'all') {
      list = list.filter((p) => {
        const c = p.confidence;
        if (confFilter === '>0.95') return c > 0.95;
        if (confFilter === '0.80-0.95') return c > 0.80 && c <= 0.95;
        if (confFilter === '0.50-0.80') return c > 0.50 && c <= 0.80;
        if (confFilter === '<=0.50') return c <= 0.50;
        return true;
      });
    }
    const sorted = [...list];
    if (sortBy === 'conf_desc') sorted.sort((a, b) => b.confidence - a.confidence);
    else if (sortBy === 'conf_asc') sorted.sort((a, b) => a.confidence - b.confidence);
    else if (sortBy === 'cluster') sorted.sort((a, b) => a.cluster_id - b.cluster_id || b.confidence - a.confidence);
    return sorted;
  }, [predictions, classFilter, confFilter, sortBy]);

  const start = page * PAGE_SIZE;
  const pageItems = filtered.slice(start, start + PAGE_SIZE);
  const lastPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1);

  return (
    <div className="flex flex-1 min-h-0 flex-col rounded-[8px] border border-[var(--border)] bg-[var(--surface)]">
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--border-muted)] p-3">
        <select
          value={classFilter}
          onChange={(e) => setClassFilter(e.target.value)}
          className="h-8 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)]"
        >
          <option value="all">All classes</option>
          {classes.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select
          value={confFilter}
          onChange={(e) => setConfFilter(e.target.value)}
          className="h-8 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)]"
        >
          {CONFIDENCE_BUCKETS.map((b) => <option key={b.value} value={b.value}>{b.label}</option>)}
        </select>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="h-8 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)]"
        >
          <option value="conf_desc">Confidence DESC</option>
          <option value="conf_asc">Confidence ASC</option>
          <option value="cluster">Cluster ID</option>
        </select>
        <span className="text-[12px] text-[var(--text-muted)]">
          {filtered.length.toLocaleString()} / {predictions.length.toLocaleString()} predictions
        </span>
        <div className="ml-auto flex items-center gap-2 text-[11px] text-[var(--text-muted)]">
          <button
            type="button"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            className="h-7 rounded-[5px] border border-[var(--border)] px-2 text-[var(--text)] hover:bg-[var(--hover)] disabled:opacity-30"
          >
            ← Prev
          </button>
          <span className="font-mono">
            page {page + 1} / {lastPage + 1}
          </span>
          <button
            type="button"
            disabled={page >= lastPage}
            onClick={() => setPage((p) => Math.min(lastPage, p + 1))}
            className="h-7 rounded-[5px] border border-[var(--border)] px-2 text-[var(--text)] hover:bg-[var(--hover)] disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {pageItems.length === 0 ? (
          <div className="flex h-full items-center justify-center text-[12px] text-[var(--text-muted)]">
            Không có prediction phù hợp với filter.
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            {pageItems.map((p) => (
              <PredictionCard key={p.result_id} prediction={p} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function PredictionCard({ prediction }) {
  const color = CLASS_COLOR[prediction.predicted_class] || 'var(--text)';
  return (
    <div className="overflow-hidden rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)]">
      <BoxThumbnail box={prediction} size={140} showLabel={false} />
      <div className="p-2 text-[11px]">
        <div className="flex items-center justify-between">
          <span className="font-semibold uppercase tracking-wide" style={{ color }}>
            {prediction.predicted_class}
          </span>
          <span
            className="rounded-full px-1.5 py-0.5 font-mono text-[10px]"
            style={{
              backgroundColor: prediction.confidence > 0.95 ? 'rgb(16 185 129 / 0.18)'
                : prediction.confidence > 0.5 ? 'rgb(251 191 36 / 0.18)'
                : 'rgb(239 68 68 / 0.18)',
              color: prediction.confidence > 0.95 ? '#10b981'
                : prediction.confidence > 0.5 ? '#fbbf24'
                : '#ef4444',
            }}
          >
            {(prediction.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="mt-1 truncate font-mono text-[10px] text-[var(--text-muted)]">
          #{prediction.result_id} · cluster {prediction.cluster_id}
        </div>
        <div className="truncate text-[10px] text-[var(--text-muted)]">{prediction.image_rel_path}</div>
      </div>
    </div>
  );
}

export default function ClassifierResults() {
  const api = typeof window !== 'undefined' ? window.electronAPI : null;
  const [paths, setPaths] = useState({ resultsDir: '', sourceDbPath: '', imageRootPath: '' });
  const [applies, setApplies] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [subTab, setSubTab] = useState('stats');

  useEffect(() => {
    if (!api) return;
    api.getClassifierResultsDefaults()
      .then((defaults) => setPaths(defaults))
      .catch((err) => setError(String(err?.message || err)));
  }, [api]);

  const loadList = useCallback(async () => {
    if (!api || !paths.resultsDir) return;
    setLoading(true);
    setError('');
    try {
      const result = await api.listClassifierRuns({ resultsDir: paths.resultsDir });
      setApplies(result.applies || []);
      if (result.applies?.length && !selectedFile) {
        setSelectedFile(result.applies[0].file);
      }
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setLoading(false);
    }
  }, [api, paths.resultsDir, selectedFile]);

  useEffect(() => { loadList(); }, [loadList]);

  useEffect(() => {
    if (!api || !selectedFile) return;
    setLoading(true);
    setError('');
    api.getClassifierApply({
      resultsDir: paths.resultsDir,
      sourceDbPath: paths.sourceDbPath,
      imageRootPath: paths.imageRootPath,
      file: selectedFile,
    })
      .then((result) => setDetails(result))
      .catch((err) => setError(String(err?.message || err)))
      .finally(() => setLoading(false));
  }, [api, selectedFile, paths.resultsDir, paths.sourceDbPath, paths.imageRootPath]);

  const apply = details?.apply;
  const metrics = details?.metrics;
  const predictions = details?.predictions || [];
  const classes = useMemo(() => metrics?.classes || apply?.class_distribution
    ? (metrics?.classes || Object.keys(apply?.class_distribution || {}))
    : [], [metrics, apply]);
  const best = metrics?.[metrics?.best_model || 'mlp'];

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden bg-[var(--bg)] p-4 text-[var(--text)]">
      <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-3">
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex flex-1 min-w-[280px] items-center gap-2 text-[12px]">
            <span className="shrink-0 text-[var(--text-muted)]">Apply run:</span>
            <select
              value={selectedFile || ''}
              onChange={(e) => setSelectedFile(e.target.value || null)}
              className="flex-1 h-8 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[12px] text-[var(--text)]"
            >
              <option value="">— select run —</option>
              {applies.map((a) => (
                <option key={a.file} value={a.file}>
                  {a.run_id.slice(0, 10)} · {a.n_predictions.toLocaleString()} preds · {formatSince(a.created_at_utc)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={loadList}
            disabled={loading}
            className="inline-flex h-8 items-center gap-1.5 rounded-[5px] border border-[var(--border)] px-3 text-[12px] font-medium text-[var(--text)] hover:bg-[var(--hover)] disabled:opacity-40"
          >
            <IconRefresh size={13} className={loading ? 'animate-spin' : ''} />
            Reload
          </button>
          {apply && (
            <span className="text-[11px] text-[var(--text-muted)]">
              Model: <span className="font-mono text-[var(--text)]">{apply.model_path?.split('/').pop()}</span>
            </span>
          )}
          {error && <span className="text-[12px] text-[var(--danger)]">{error}</span>}
        </div>
      </div>

      {!details ? (
        <div className="flex flex-1 items-center justify-center text-[12px] text-[var(--text-muted)]">
          {loading ? 'Loading…' : 'Chọn 1 apply run ở trên để xem kết quả.'}
        </div>
      ) : (
        <>
          <div className="inline-flex items-center gap-1 rounded-[8px] border border-[var(--border)] bg-[var(--surface)] p-1 text-[12px]">
            <button
              type="button"
              onClick={() => setSubTab('stats')}
              className={`inline-flex items-center gap-1.5 rounded-[6px] px-3 py-1.5 font-medium transition-colors ${
                subTab === 'stats'
                  ? 'bg-[var(--primary)] text-white'
                  : 'text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'
              }`}
            >
              <IconChartHistogram size={14} />
              Thống kê
            </button>
            <button
              type="button"
              onClick={() => setSubTab('predictions')}
              className={`inline-flex items-center gap-1.5 rounded-[6px] px-3 py-1.5 font-medium transition-colors ${
                subTab === 'predictions'
                  ? 'bg-[var(--primary)] text-white'
                  : 'text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'
              }`}
            >
              <IconLayoutGrid size={14} />
              Predictions
              <span className="rounded-full bg-current/15 px-1.5 py-0.5 text-[10px] font-mono opacity-90">
                {predictions.length.toLocaleString()}
              </span>
            </button>
          </div>

          {subTab === 'stats' ? (
            <div className="flex flex-1 min-h-0 flex-col gap-3 overflow-y-auto">
              <MetricsCards apply={apply} metrics={metrics} />
              <div className="grid gap-3 lg:grid-cols-2">
                <ConfusionMatrix
                  classes={classes}
                  matrix={best?.confusion_matrix}
                  title={`Confusion matrix (${metrics?.best_model || 'best'} test split)`}
                />
                <PerClassMetrics
                  classes={classes}
                  report={best?.report}
                  title="Per-class metrics"
                />
              </div>
              <PredictionDistribution classDist={apply?.class_distribution} total={apply?.n_predictions || 0} />
            </div>
          ) : (
            <PredictionsBrowser predictions={predictions} classes={classes} />
          )}
        </>
      )}
    </div>
  );
}
