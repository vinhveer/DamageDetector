import { useEffect, useMemo, useState } from 'react';
import { IconChevronLeft, IconStar, IconPlus } from '@tabler/icons-react';
import { Button, EmptyState, ErrorMessage, IconButton, SelectControl } from '../../components/ui/index.js';
import PageHeader from '../shared/PageHeader.jsx';
import { cn } from '../../components/ui/cn.js';
import { formatFloat, formatNumber, shortId } from '../shared/viewerUtils.js';

const BUCKETS = [
  { value: 'all', label: 'All' },
  { value: 'auto_accept', label: 'Auto accept' },
  { value: 'need_review', label: 'Need review' },
  { value: 'ambiguous', label: 'Ambiguous' },
  { value: 'mixed', label: 'Mixed' },
  { value: 'unknown', label: 'Unknown' },
  { value: 'label_conflict', label: 'Conflict' },
  { value: 'prototype', label: 'Prototype' },
];

const bucketClass = (bucket) => {
  if (bucket === 'auto_accept') return 'border-[var(--success)] bg-[var(--success-bg)] text-[var(--success)]';
  if (bucket === 'prototype') return 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]';
  if (bucket === 'label_conflict') return 'border-[var(--danger)] bg-[var(--danger-bg)] text-[var(--danger)]';
  if (bucket === 'ambiguous' || bucket === 'mixed') return 'border-[var(--warning)] bg-[var(--warning-bg)] text-[var(--warning)]';
  return 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]';
};

function QualityTab({ scores, run }) {
  const thresholds = useMemo(() => {
    try { return JSON.parse(run.thresholds_json || '{}'); } catch { return {}; }
  }, [run.thresholds_json]);
  const autoThreshold = thresholds.auto_threshold ?? 0.78;

  const metrics = useMemo(() => {
    const labels = ['crack', 'spall', 'mold'];
    return labels.map((label) => {
      const clusters = scores.filter((s) => s.recommended_label === label);
      if (clusters.length === 0) return { label, cohesion: 0, separation: 0, coverage: 0, count: 0 };
      const cohesion = clusters.reduce((sum, s) => sum + s.top_score, 0) / clusters.length;
      const separation = clusters.reduce((sum, s) => sum + s.confidence_gap, 0) / clusters.length;
      const coverage = clusters.filter((s) => s.top_score >= autoThreshold).length / clusters.length;
      return { label, cohesion, separation, coverage, count: clusters.length };
    });
  }, [scores, autoThreshold]);

  const verdict = (m) => m.cohesion >= 0.75 && m.separation >= 0.10 && m.coverage >= 0.80;

  const lowConfidence = useMemo(() => scores.filter((s) => s.confidence_gap < 0.05), [scores]);

  return (
    <div className="p-6 space-y-6">
      <table className="w-full max-w-[700px] border-collapse text-[13px]">
        <thead className="text-[12px] font-semibold text-[var(--text-muted)]">
          <tr>
            <th className="px-4 py-2 text-left">Label</th>
            <th className="px-4 py-2 text-right">Clusters</th>
            <th className="px-4 py-2 text-right">Cohesion</th>
            <th className="px-4 py-2 text-right">Separation</th>
            <th className="px-4 py-2 text-right">Coverage</th>
            <th className="px-4 py-2 text-center">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={m.label} className="border-t border-[var(--border-muted)]">
              <td className="px-4 py-2.5 font-medium">{m.label}</td>
              <td className="px-4 py-2.5 text-right tabular-nums">{m.count}</td>
              <td className="px-4 py-2.5 text-right tabular-nums">{formatFloat(m.cohesion, 3)}</td>
              <td className="px-4 py-2.5 text-right tabular-nums">{formatFloat(m.separation, 3)}</td>
              <td className="px-4 py-2.5 text-right tabular-nums">{Math.round(m.coverage * 100)}%</td>
              <td className="px-4 py-2.5 text-center">
                {m.count === 0 ? '—' : verdict(m) ? <span className="text-[var(--success)]">✓ Stable</span> : <span className="text-[var(--warning)]">⚠ Needs work</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {lowConfidence.length > 0 && (
        <div>
          <h3 className="mb-2 text-[13px] font-semibold text-[var(--text)]">
            {lowConfidence.length} clusters with confidence_gap &lt; 0.05
          </h3>
          <div className="max-h-[200px] overflow-auto rounded border border-[var(--border-muted)] text-[12px]">
            {lowConfidence.slice(0, 50).map((s) => (
              <div key={s.cluster_key} className="flex items-center gap-3 border-b border-[var(--border-muted)] px-3 py-1.5">
                <span className="font-medium">{s.cluster_key}</span>
                <span className="text-[var(--text-muted)]">{s.recommended_label}</span>
                <span className="ml-auto tabular-nums">gap {formatFloat(s.confidence_gap, 4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function VersionDetail({ paths, reviewRunId, onBack, onNewVersion }) {
  const [run, setRun] = useState(null);
  const [scores, setScores] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tab, setTab] = useState('clusters');
  const [bucketFilter, setBucketFilter] = useState('all');
  const [labelFilter, setLabelFilter] = useState('all');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    window.electronAPI.getPrototypeVersionDetail({ reviewDbPath: paths.reviewDbPath, review_run_id: reviewRunId })
      .then((result) => {
        if (cancelled) return;
        if (result.error) { setError(result.error); return; }
        setRun(result.run);
        setScores(result.scores || []);
      })
      .catch((e) => { if (!cancelled) setError(String(e.message || e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [reviewRunId, paths.reviewDbPath]);

  const filteredScores = useMemo(() => {
    let list = scores;
    if (bucketFilter !== 'all') list = list.filter((s) => s.review_bucket === bucketFilter);
    if (labelFilter !== 'all') list = list.filter((s) => s.recommended_label === labelFilter);
    return list;
  }, [scores, bucketFilter, labelFilter]);

  if (loading) return <EmptyState title="Loading" />;
  if (!run) return <EmptyState title="Version not found" />;

  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title={run.display_name || shortId(run.review_run_id)}
        subtitle={`${run.model_name} · ${formatNumber(run.total_clusters)} clusters`}
        left={<IconButton label="Back" onClick={onBack}><IconChevronLeft size={16} /></IconButton>}
        right={
          <div className="flex items-center gap-2">
            {run.is_active ? <span className="flex items-center gap-1 text-[12px] text-[var(--warning)]"><IconStar size={12} /> active</span> : null}
            <Button onClick={() => onNewVersion(reviewRunId)}><IconPlus size={14} /> New from this</Button>
          </div>
        }
      />
      {error && <div className="px-6 py-2"><ErrorMessage error={error} /></div>}

      {/* Tabs */}
      <div className="flex items-center gap-4 border-b border-[var(--border-muted)] px-6">
        {['clusters', 'quality'].map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={cn(
              'border-b-2 px-1 py-2.5 text-[13px] font-medium capitalize',
              tab === t ? 'border-[var(--primary)] text-[var(--text)]' : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text)]'
            )}
          >
            {t}
          </button>
        ))}
        {tab === 'clusters' && (
          <div className="ml-auto flex items-center gap-2">
            <SelectControl value={bucketFilter} onChange={(e) => setBucketFilter(e.currentTarget.value)} className="w-[130px]">
              {BUCKETS.map((b) => <option key={b.value} value={b.value}>{b.label}</option>)}
            </SelectControl>
            <SelectControl value={labelFilter} onChange={(e) => setLabelFilter(e.currentTarget.value)} className="w-[110px]">
              <option value="all">All labels</option>
              <option value="crack">crack</option>
              <option value="spall">spall</option>
              <option value="mold">mold</option>
            </SelectControl>
            <span className="text-[12px] text-[var(--text-muted)]">{filteredScores.length} groups</span>
          </div>
        )}
      </div>

      <main className="min-h-0 flex-1 overflow-auto">
        {tab === 'quality' && <QualityTab scores={scores} run={run} />}
        {tab === 'clusters' && (
          <table className="w-full min-w-[1000px] border-collapse text-left text-[13px]">
            <thead className="sticky top-0 z-10 border-b border-[var(--border-muted)] bg-[var(--surface)] text-[12px] font-semibold text-[var(--text-muted)]">
              <tr>
                <th className="px-6 py-3">Group</th>
                <th className="w-28 px-4 py-3">Original</th>
                <th className="w-28 px-4 py-3">Suggested</th>
                <th className="w-28 px-4 py-3">Bucket</th>
                <th className="w-20 px-4 py-3 text-right">Top</th>
                <th className="w-20 px-4 py-3 text-right">Gap</th>
                <th className="w-20 px-4 py-3 text-right">Purity</th>
                <th className="w-20 px-4 py-3 text-right">Size</th>
                <th className="px-4 py-3">Reason</th>
              </tr>
            </thead>
            <tbody>
              {filteredScores.map((s) => (
                <tr key={s.cluster_key} className="border-b border-[var(--border-muted)] hover:bg-[var(--hover)]">
                  <td className="px-6 py-3 font-medium">{s.cluster_key}</td>
                  <td className="px-4 py-3 text-[var(--text-muted)]">{s.original_major_label}</td>
                  <td className="px-4 py-3 font-medium">{s.recommended_label}</td>
                  <td className="px-4 py-3">
                    <span className={cn('rounded-[4px] border px-1.5 py-0.5 text-[11px] font-medium', bucketClass(s.review_bucket))}>
                      {s.review_bucket}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">{formatFloat(s.top_score, 3)}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{formatFloat(s.confidence_gap, 3)}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{formatFloat(s.purity, 2)}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{s.cluster_size}</td>
                  <td className="px-4 py-3 text-[var(--text-muted)]">{s.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </main>
    </div>
  );
}
