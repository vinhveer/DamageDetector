import { useEffect, useMemo, useState } from 'react';
import { IconChevronLeft, IconRefresh } from '@tabler/icons-react';
import { Button, EmptyState, ErrorMessage, IconButton, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';
import PageHeader from '../resultViewer/components/PageHeader.jsx';
import ImageGrid from '../resultViewer/components/ImageGrid.jsx';
import { formatFloat, formatNumber, groupRowsByImage, shortId } from '../resultViewer/utils.js';

const BUCKETS = [
  { value: 'all', label: 'All' },
  { value: 'label_conflict', label: 'Conflict' },
  { value: 'ambiguous', label: 'Ambiguous' },
  { value: 'mixed', label: 'Mixed' },
  { value: 'unknown', label: 'Unknown' },
  { value: 'need_review', label: 'Need review' },
  { value: 'prototype', label: 'Prototype' },
  { value: 'auto_accept', label: 'Auto accept' }
];

const LABELS = [
  { value: 'all', label: 'All labels' },
  { value: 'crack', label: 'crack' },
  { value: 'spall', label: 'spall' },
  { value: 'mold', label: 'mold' }
];

const bucketClass = (bucket) => {
  if (bucket === 'auto_accept') return 'border-[var(--success)] bg-[var(--success-bg)] text-[var(--success)]';
  if (bucket === 'prototype') return 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]';
  if (bucket === 'label_conflict') return 'border-[var(--danger)] bg-[var(--danger-bg)] text-[var(--danger)]';
  if (bucket === 'ambiguous' || bucket === 'mixed') return 'border-[var(--warning)] bg-[var(--warning-bg)] text-[var(--warning)]';
  return 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)]';
};

const scoreClass = (label, suggested) => cn('tabular-nums', label === suggested ? 'font-semibold text-[var(--text)]' : 'text-[var(--text-muted)]');

export default function PrototypeReview() {
  const [paths, setPaths] = useState({ reviewDbPath: '', imageRootPath: '' });
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [scores, setScores] = useState([]);
  const [bucket, setBucket] = useState('all');
  const [label, setLabel] = useState('all');
  const [query, setQuery] = useState('');
  const [selectedScore, setSelectedScore] = useState(null);
  const [assignments, setAssignments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');

  const selectedRun = runs.find((run) => run.review_run_id === selectedRunId) || null;
  const imageGroups = useMemo(() => groupRowsByImage(assignments), [assignments]);
  const visibleScores = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return scores;
    return scores.filter((item) => [
      item.cluster_key,
      item.original_major_label,
      item.recommended_label,
      item.review_bucket,
      item.reason
    ].some((value) => String(value || '').toLowerCase().includes(needle)));
  }, [scores, query]);

  const loadRuns = async (nextPaths = paths) => {
    if (!nextPaths.reviewDbPath) return;
    setLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listPrototypeReviewRuns({ reviewDbPath: nextPaths.reviewDbPath });
      const nextRuns = result.runs || [];
      setRuns(nextRuns);
      setSelectedRunId(nextRuns[0]?.review_run_id || '');
      setSelectedScore(null);
      setAssignments([]);
    } catch (event) {
      setError(String(event.message || event));
      setRuns([]);
      setSelectedRunId('');
    } finally {
      setLoading(false);
    }
  };

  const loadScores = async () => {
    if (!selectedRunId || !paths.reviewDbPath) return;
    setLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listPrototypeReviewScores({
        reviewDbPath: paths.reviewDbPath,
        reviewRunId: selectedRunId,
        bucket,
        label
      });
      setScores(result.scores || []);
      setSelectedScore(null);
      setAssignments([]);
    } catch (event) {
      setError(String(event.message || event));
      setScores([]);
    } finally {
      setLoading(false);
    }
  };

  const openScore = async (score) => {
    if (!selectedRun) return;
    setSelectedScore(score);
    setDetailLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listPrototypeReviewAssignments({
        reviewDbPath: paths.reviewDbPath,
        reviewRunId: selectedRun.review_run_id,
        featureDbPath: selectedRun.feature_db_path,
        sourceDbPath: selectedRun.source_db_path,
        imageRootPath: paths.imageRootPath,
        groupingRunId: selectedRun.grouping_run_id,
        clusterKey: score.cluster_key
      });
      setAssignments(result.assignments || []);
    } catch (event) {
      setError(String(event.message || event));
      setAssignments([]);
    } finally {
      setDetailLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    window.electronAPI.getPrototypeReviewDefaults()
      .then((defaults) => {
        if (cancelled) return;
        const nextPaths = {
          reviewDbPath: defaults.reviewDbPath || '',
          imageRootPath: defaults.imageRootPath || ''
        };
        setPaths(nextPaths);
        loadRuns(nextPaths);
      })
      .catch((event) => {
        if (!cancelled) setError(String(event.message || event));
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    loadScores();
  }, [selectedRunId, bucket, label]);

  if (selectedScore) {
    return (
      <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
        <PageHeader
          title={selectedScore.cluster_key}
          subtitle={`${selectedScore.original_major_label} -> ${selectedScore.recommended_label} · ${selectedScore.review_bucket}`}
          left={
            <IconButton label="Back" onClick={() => { setSelectedScore(null); setAssignments([]); }}>
              <IconChevronLeft size={16} />
            </IconButton>
          }
          right={
            <div className="text-[12px] text-[var(--text-muted)]">
              score {formatFloat(selectedScore.top_score, 3)} · gap {formatFloat(selectedScore.confidence_gap, 3)} · vote {formatFloat(selectedScore.crop_vote_ratio, 2)}
            </div>
          }
        />
        {error && <div className="px-8 py-3"><ErrorMessage error={error} /></div>}
        <main className="min-h-0 flex-1 overflow-hidden bg-[var(--bg)]">
          {detailLoading ? <EmptyState title="Loading" /> : <ImageGrid groups={imageGroups} imageSize={220} onOpenImage={() => {}} />}
        </main>
      </div>
    );
  }

  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title="Prototype Review"
        subtitle={selectedRun ? `${formatNumber(selectedRun.total_clusters)} groups · ${shortId(selectedRun.review_run_id)}` : 'Run Step 5 to generate prototype_review.sqlite3'}
        right={
          <>
            <SelectControl value={selectedRunId} onChange={(event) => setSelectedRunId(event.currentTarget.value)} className="w-[280px]" disabled={runs.length === 0}>
              <option value="">No review run</option>
              {runs.map((run) => (
                <option key={run.review_run_id} value={run.review_run_id}>
                  {run.created_at_utc?.slice(0, 16) || 'Unknown'} · {shortId(run.review_run_id)}
                </option>
              ))}
            </SelectControl>
            <Button onClick={() => loadRuns()}>
              <IconRefresh size={14} />
              Reload
            </Button>
          </>
        }
      />
      <div className="flex min-h-14 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--bg)] px-6 py-3">
        <SelectControl value={bucket} onChange={(event) => setBucket(event.currentTarget.value)} className="w-[150px]">
          {BUCKETS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </SelectControl>
        <SelectControl value={label} onChange={(event) => setLabel(event.currentTarget.value)} className="w-[140px]">
          {LABELS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </SelectControl>
        <TextInput value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="Filter groups" className="w-[280px]" />
        <div className="ml-auto text-[13px] text-[var(--text-muted)]">{formatNumber(visibleScores.length)} groups</div>
      </div>
      {error && <div className="px-8 py-3"><ErrorMessage error={error} /></div>}
      <main className="min-h-0 flex-1 overflow-auto bg-[var(--bg)]">
        {loading && <EmptyState title="Loading" />}
        {!loading && runs.length === 0 && <EmptyState title="No Step 5 review DB" />}
        {!loading && runs.length > 0 && (
          <table className="w-full min-w-[1120px] border-collapse text-left text-[13px]">
            <thead className="sticky top-0 z-10 border-b border-[var(--border-muted)] bg-[var(--surface)] text-[12px] font-semibold text-[var(--text-muted)]">
              <tr>
                <th className="px-6 py-3.5 text-left">Group</th>
                <th className="w-28 px-4 py-3.5 text-left">Original</th>
                <th className="w-28 px-4 py-3.5 text-left">Suggested</th>
                <th className="w-32 px-4 py-3.5 text-left">Bucket</th>
                <th className="w-24 px-4 py-3.5 text-right">Top</th>
                <th className="w-24 px-4 py-3.5 text-right">Gap</th>
                <th className="w-24 px-4 py-3.5 text-right">Vote</th>
                <th className="w-40 px-4 py-3.5 text-right">Scores</th>
                <th className="w-24 px-4 py-3.5 text-right">Rows</th>
                <th className="px-4 py-3.5 text-left">Reason</th>
              </tr>
            </thead>
            <tbody>
              {visibleScores.map((score) => (
                <tr key={score.cluster_key} onClick={() => openScore(score)} className="cursor-pointer border-b border-[var(--border-muted)] hover:bg-[var(--hover)]">
                  <td className="px-6 py-4 font-medium text-[var(--text)]">{score.cluster_key}</td>
                  <td className="px-4 py-4 text-[var(--text-muted)]">{score.original_major_label}</td>
                  <td className="px-4 py-4 font-medium text-[var(--text)]">{score.recommended_label}</td>
                  <td className="px-4 py-4">
                    <span className={cn('rounded-[4px] border px-1.5 py-0.5 text-[11px] font-medium', bucketClass(score.review_bucket))}>{score.review_bucket}</span>
                  </td>
                  <td className="px-4 py-4 text-right tabular-nums text-[var(--text)]">{formatFloat(score.top_score, 3)}</td>
                  <td className="px-4 py-4 text-right tabular-nums text-[var(--text)]">{formatFloat(score.confidence_gap, 3)}</td>
                  <td className="px-4 py-4 text-right tabular-nums text-[var(--text)]">{formatFloat(score.crop_vote_ratio, 2)}</td>
                  <td className="px-4 py-4 text-right text-[12px]">
                    <span className={scoreClass('crack', score.recommended_label)}>C {formatFloat(score.score_crack, 2)}</span>
                    <span className="text-[var(--text-muted)]"> · </span>
                    <span className={scoreClass('spall', score.recommended_label)}>S {formatFloat(score.score_spall, 2)}</span>
                    <span className="text-[var(--text-muted)]"> · </span>
                    <span className={scoreClass('mold', score.recommended_label)}>M {formatFloat(score.score_mold, 2)}</span>
                  </td>
                  <td className="px-4 py-4 text-right tabular-nums text-[var(--text)]">{formatNumber(score.cluster_size)}</td>
                  <td className="px-4 py-4 text-[var(--text-muted)]">{score.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!loading && runs.length > 0 && visibleScores.length === 0 && <EmptyState title="No matching groups" />}
      </main>
    </div>
  );
}
