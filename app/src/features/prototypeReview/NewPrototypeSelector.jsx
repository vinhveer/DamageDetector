import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconChevronLeft, IconCheck, IconAlertTriangle, IconX } from '@tabler/icons-react';
import { Button, EmptyState, ErrorMessage, IconButton, TextInput } from '../../components/ui/index.js';
import PageHeader from '../shared/PageHeader.jsx';
import { cn } from '../../components/ui/cn.js';
import { formatFloat } from '../shared/viewerUtils.js';
import { imageLoaderPool } from '../../utils/imageLoaderPool.js';
import FilterBar from './FilterBar.jsx';

const LABELS = ['crack', 'spall', 'mold'];
const TABS = ['crack', 'spall', 'mold', 'excluded'];
const SORT_OPTIONS = [
  { value: 'top_score', label: 'Top score' },
  { value: 'confidence_gap', label: 'Confidence gap' },
  { value: 'purity', label: 'Purity' },
  { value: 'cluster_size', label: 'Cluster size' },
  { value: 'recently_picked', label: 'Recently picked' },
  { value: 'cluster_key', label: 'Cluster key' },
];
const ALL_REVIEW_BUCKETS = ['auto_accept', 'prototype', 'need_review', 'unknown', 'ambiguous', 'mixed', 'label_conflict'];
const BATCH_DELAY = 80;
const LABEL_FILTERS = ['crack', 'spall', 'mold'];

const emptyBucket = () => ({ clusters: new Set(), images: new Set() });
const defaultFilters = () => ({
  search: '',
  buckets: [...ALL_REVIEW_BUCKETS],
  labels: [...LABEL_FILTERS],
  picks: 'all',
  minTopScore: 0,
  minConfidenceGap: 0,
});

// ── Bucket badge color map (8.1) ──────────────────────────────────────────────

const BUCKET_BADGE_COLORS = {
  auto_accept: 'bg-green-600 text-white',
  prototype: 'bg-blue-600 text-white',
  need_review: 'bg-amber-500 text-white',
  unknown: 'bg-gray-500 text-white',
  ambiguous: 'bg-orange-500 text-white',
  mixed: 'bg-purple-600 text-white',
  label_conflict: 'bg-red-600 text-white',
  excluded: 'bg-red-900 text-white',
};

// ── CroppedThumb (8.6) ───────────────────────────────────────────────────────
// state: 'unpicked' | 'picked' | 'picked-other' | 'excluded' | 'group-picked'
// otherLabel?: string — label name when state is 'picked-other'

function CroppedThumb({ row, size = 120, priority = 1, state = 'unpicked', otherLabel, onClick, onDoubleClick }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!row.image_uri) return;
    const task = imageLoaderPool.enqueue({
      uri: row.image_uri,
      bbox: [row.x1, row.y1, row.x2, row.y2],
      size,
      priority,
      onBitmap: (bitmap) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, size, size);
        ctx.drawImage(bitmap, 0, 0);
      },
      onError: () => {},
    });
    return () => task.cancel();
  }, [priority, row.image_uri, row.x1, row.y1, row.x2, row.y2, size]);

  const borderClass = {
    unpicked: 'border-[var(--border-muted)]',
    picked: 'border-[var(--primary-dark,var(--primary))]',
    'picked-other': 'border-[var(--primary)] opacity-70',
    excluded: 'border-[var(--danger)] opacity-40',
    'group-picked': 'border-[var(--primary)] bg-[var(--primary-bg)]',
  }[state] || 'border-[var(--border-muted)]';

  return (
    <button
      type="button"
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      className={cn(
        'relative flex-shrink-0 cursor-pointer overflow-hidden rounded-[4px] border-2',
        borderClass
      )}
      style={{ width: size, height: size }}
    >
      <canvas ref={canvasRef} width={size} height={size} className="block" />
      {state === 'excluded' && (
        <div className="absolute inset-0 flex items-center justify-center bg-[rgba(0,0,0,0.4)]">
          <IconX size={32} className="text-[var(--danger)]" />
        </div>
      )}
      {state === 'picked' && (
        <div className="absolute right-1 top-1 flex h-[14px] w-[14px] items-center justify-center rounded-full bg-[var(--primary)]">
          <IconCheck size={8} className="text-white" />
        </div>
      )}
      {state === 'picked-other' && otherLabel && (
        <span className="absolute left-1 top-1 rounded bg-[var(--primary-bg)] px-1 text-[9px] font-medium text-[var(--primary)]">
          {otherLabel}
        </span>
      )}
    </button>
  );
}

// ── Gallery Row ───────────────────────────────────────────────────────────────

function CandidateGalleryRow({ score, assignments, loadingRow, picked, excluded, activeLabel, selections, onAssignCluster, onUnassignCluster, onAssignImage, onUnassignImage, onRequestAssignments }) {
  const rowRef = useRef(null);
  const [thumbPriority, setThumbPriority] = useState(2);
  const isRelabel = score.current_label !== activeLabel;

  const visibleAssignments = useMemo(
    () => (assignments || []),
    [assignments]
  );

  useEffect(() => {
    const preloadObserver = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        onRequestAssignments(score.cluster_key);
        setThumbPriority((p) => Math.min(p, 2));
        preloadObserver.disconnect();
      }
    }, { rootMargin: '600px 0px' });
    const visibleObserver = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) setThumbPriority(1);
    }, { rootMargin: '0px' });
    if (rowRef.current) {
      preloadObserver.observe(rowRef.current);
      visibleObserver.observe(rowRef.current);
    }
    return () => {
      preloadObserver.disconnect();
      visibleObserver.disconnect();
    };
  }, [onRequestAssignments, score.cluster_key]);

  // 8.2: Determine cluster assignment badge
  const assignmentBadge = useMemo(() => {
    // Check if cluster is group-picked in any tab
    for (const l of TABS) {
      if (selections[l].clusters.has(score.cluster_key)) {
        if (l === 'excluded') return { text: '→ excluded', color: 'text-[var(--danger)]' };
        return { text: `→ ${l} (group)`, color: 'text-[var(--primary)]' };
      }
    }
    // Check if some images from this cluster are individually picked
    const clusterAssignments = assignments || [];
    let hasImagePick = false;
    for (const a of clusterAssignments) {
      const rid = Number(a.result_id);
      for (const l of TABS) {
        if (selections[l].images.has(rid)) { hasImagePick = true; break; }
      }
      if (hasImagePick) break;
    }
    if (hasImagePick) return { text: 'mixed picks', color: 'text-[var(--text-muted)]' };
    return null;
  }, [selections, score.cluster_key, assignments]);

  // 8.6: Determine thumbnail state
  const getThumbState = useCallback((resultId) => {
    // Check excluded first
    if (selections.excluded.images.has(resultId)) return { state: 'excluded' };
    // Check if picked into current active label (image-level)
    if (selections[activeLabel].images.has(resultId)) return { state: 'picked' };
    // Check if picked into another label (image-level or group-level)
    for (const l of TABS) {
      if (l === activeLabel || l === 'excluded') continue;
      if (selections[l].images.has(resultId)) return { state: 'picked-other', otherLabel: l };
      if (selections[l].clusters.has(score.cluster_key)) return { state: 'picked-other', otherLabel: l };
    }
    // Check if part of group-picked cluster for current label
    if (picked) return { state: 'group-picked' };
    return { state: 'unpicked' };
  }, [selections, activeLabel, picked, score.cluster_key]);

  return (
    <div ref={rowRef} className={cn(
      'border-b border-[var(--border-muted)] px-6 py-3',
      picked && 'bg-[var(--primary-bg)]',
      excluded && 'bg-[var(--danger-bg)] opacity-60'
    )}>
      {/* Header (8.1 + 8.2) */}
      <div className="mb-2 flex items-center gap-3">
        <button
          type="button"
          onClick={() => picked ? onUnassignCluster(score.cluster_key) : onAssignCluster(score.cluster_key, activeLabel)}
          className={cn(
            'flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full border-2',
            picked ? 'border-[var(--primary)] bg-[var(--primary)] text-white' : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--primary)]'
          )}
        >
          {picked && <IconCheck size={12} />}
        </button>
        <span className="text-[13px] font-semibold text-[var(--text)]">{score.cluster_key}</span>
        <span className="text-[12px] text-[var(--text-muted)]">
          {score.current_label} · top {formatFloat(score.top_score, 3)} · {score.cluster_size} imgs
        </span>
        {/* 8.1: review_bucket badge */}
        {score.review_bucket && (
          <span className={cn(
            'rounded-full px-2 py-0.5 text-[10px] font-medium',
            BUCKET_BADGE_COLORS[score.review_bucket] || 'bg-gray-500 text-white'
          )}>
            {score.review_bucket}
          </span>
        )}
        {isRelabel && (
          <span className="flex items-center gap-1 text-[11px] text-[var(--warning)]">
            <IconAlertTriangle size={11} /> → {activeLabel}
          </span>
        )}
        {/* 8.2: Assignment badge on right */}
        {assignmentBadge && (
          <span className={cn('ml-auto text-[11px] font-medium', assignmentBadge.color)}>
            {assignmentBadge.text}
          </span>
        )}
      </div>
      {/* Thumbnail strip */}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {loadingRow && <div className="flex h-[120px] items-center text-[12px] text-[var(--text-muted)]">Loading…</div>}
        {visibleAssignments.map((row) => {
          const resultId = Number(row.result_id);
          const thumbInfo = getThumbState(resultId);
          return (
            <CroppedThumb
              key={row.result_id}
              row={row}
              size={120}
              priority={thumbPriority}
              state={thumbInfo.state}
              otherLabel={thumbInfo.otherLabel}
              onClick={() => {
                // 8.4: Thumbnail click behavior
                if (picked) {
                  // Cluster is group-picked for current label → ungroup, then pick this image
                  onUnassignCluster(score.cluster_key);
                  onAssignImage(resultId, activeLabel);
                } else {
                  // Toggle image pick for current label
                  if (selections[activeLabel].images.has(resultId)) {
                    onUnassignImage(resultId);
                  } else {
                    onAssignImage(resultId, activeLabel);
                  }
                }
              }}
              onDoubleClick={() => onAssignImage(resultId, 'excluded')}
            />
          );
        })}
        {assignments && visibleAssignments.length === 0 && !loadingRow && (
          <div className="text-[12px] text-[var(--text-muted)]">No images</div>
        )}
      </div>
    </div>
  );
}

// ── Batched assignment loader ─────────────────────────────────────────────────

function useBatchedLoader(parentRun, paths, enabled) {
  const [assignmentsByKey, setAssignmentsByKey] = useState({});
  const [loadingKeys, setLoadingKeys] = useState(new Set());
  const pendingRef = useRef(new Set());
  const requestedRef = useRef(new Set());
  const timerRef = useRef(null);

  const flush = useCallback(() => {
    timerRef.current = null;
    const keys = Array.from(pendingRef.current);
    pendingRef.current.clear();
    if (!enabled || !parentRun || !keys.length) return;

    window.electronAPI.listPrototypeReviewAssignmentsBulk({
      reviewDbPath: paths.reviewDbPath,
      reviewRunId: parentRun.review_run_id,
      featureDbPath: parentRun.feature_db_path,
      sourceDbPath: parentRun.source_db_path,
      imageRootPath: paths.imageRootPath,
      groupingRunId: parentRun.grouping_run_id,
      clusterKeys: keys
    }).then((result) => {
      const byKey = result.assignmentsByClusterKey || {};
      setAssignmentsByKey((prev) => {
        const next = { ...prev };
        for (const k of keys) next[k] = byKey[k] || [];
        return next;
      });
    }).catch(() => {
      setAssignmentsByKey((prev) => {
        const next = { ...prev };
        for (const k of keys) next[k] = [];
        return next;
      });
    }).finally(() => {
      setLoadingKeys((prev) => {
        const next = new Set(prev);
        for (const k of keys) next.delete(k);
        return next;
      });
    });
  }, [enabled, parentRun, paths]);

  const request = useCallback((key) => {
    if (!enabled || !parentRun || requestedRef.current.has(key)) return;
    requestedRef.current.add(key);
    pendingRef.current.add(key);
    setLoadingKeys((prev) => new Set(prev).add(key));
    if (!timerRef.current) timerRef.current = setTimeout(flush, BATCH_DELAY);
  }, [enabled, parentRun, flush]);

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  return { assignmentsByKey, loadingKeys, request };
}

// ── Main component ────────────────────────────────────────────────────────────

export default function NewPrototypeSelector({ paths, parentReviewRunId, onBack, onCreated }) {
  const [displayName, setDisplayName] = useState('');
  const [setActive, setSetActive] = useState(true);
  const [activeLabel, setActiveLabel] = useState('crack');
  const [selections, setSelections] = useState({
    crack: emptyBucket(),
    spall: emptyBucket(),
    mold: emptyBucket(),
    excluded: emptyBucket(),
  });
  const [candidates, setCandidates] = useState([]);
  const [parentRun, setParentRun] = useState(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [sort, setSort] = useState('top_score');
  const [filters, setFilters] = useState(defaultFilters);
  const [pickedAt, setPickedAt] = useState({ clusters: {}, images: {} });
  const [showLoaderStats, setShowLoaderStats] = useState(false);
  const [loaderStats, setLoaderStats] = useState(() => imageLoaderPool.stats());
  const [jobSessionId, setJobSessionId] = useState(null);
  const [jobLog, setJobLog] = useState('');

  useEffect(() => {
    setFilters(defaultFilters());
    setSort('top_score');
    setSelections({
      crack: emptyBucket(),
      spall: emptyBucket(),
      mold: emptyBucket(),
      excluded: emptyBucket(),
    });
    setPickedAt({ clusters: {}, images: {} });
  }, [parentReviewRunId]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === 'i') {
        setShowLoaderStats((value) => !value);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    if (!showLoaderStats) return undefined;
    const timer = window.setInterval(() => setLoaderStats(imageLoaderPool.stats()), 500);
    return () => window.clearInterval(timer);
  }, [showLoaderStats]);

  // Load parent run info for assignment fetching
  useEffect(() => {
    if (!parentReviewRunId) return;
    window.electronAPI.getPrototypeVersionDetail({ reviewDbPath: paths.reviewDbPath, review_run_id: parentReviewRunId })
      .then((r) => { if (r.run) setParentRun(r.run); })
      .catch(() => {});
  }, [parentReviewRunId, paths.reviewDbPath]);

  const { assignmentsByKey, loadingKeys, request: requestAssignments } = useBatchedLoader(parentRun, paths, !!parentRun);

  const loadCandidates = useCallback(async () => {
    if (!parentReviewRunId) return;
    setLoading(true);
    try {
      const result = await window.electronAPI.getPrototypeCandidates({
        reviewDbPath: paths.reviewDbPath,
        parent_review_run_id: parentReviewRunId,
        filters: {},
        sort,
        thumbnails_per_cluster: -1
      });
      setCandidates(result.candidates || []);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }, [parentReviewRunId, paths.reviewDbPath, sort]);

  useEffect(() => { loadCandidates(); }, [loadCandidates]);

  // Job events
  useEffect(() => {
    if (!jobSessionId) return;
    return window.electronAPI.onPrototypeReviewJobEvent((ev) => {
      if (ev.sessionId !== jobSessionId) return;
      if (ev.type === 'stdout' || ev.type === 'stderr') setJobLog((prev) => prev + ev.data);
      if (ev.type === 'closed') {
        setSubmitting(false);
        if (ev.data === '0') {
          const match = jobLog.match(/review_run_id=([a-f0-9]+)/);
          onCreated(match?.[1] || null);
        } else {
          setError('Job failed. Check log above.');
        }
      }
    });
  }, [jobSessionId, jobLog, onCreated]);

  // ── 6.4: Mutual exclusivity helpers ──────────────────────────────────────────

  const assignCluster = useCallback((clusterKey, targetLabel) => {
    setSelections(prev => {
      const next = {};
      for (const l of TABS) {
        next[l] = { clusters: new Set(prev[l].clusters), images: new Set(prev[l].images) };
        next[l].clusters.delete(clusterKey); // remove from all
      }
      next[targetLabel].clusters.add(clusterKey);
      return next;
    });
    setPickedAt((prev) => ({
      ...prev,
      clusters: { ...prev.clusters, [clusterKey]: Date.now() },
    }));
  }, []);

  const assignImage = useCallback((resultId, targetLabel) => {
    setSelections(prev => {
      const next = {};
      for (const l of TABS) {
        next[l] = { clusters: new Set(prev[l].clusters), images: new Set(prev[l].images) };
        next[l].images.delete(resultId); // remove from all
      }
      next[targetLabel].images.add(resultId);
      return next;
    });
    setPickedAt((prev) => ({
      ...prev,
      images: { ...prev.images, [resultId]: Date.now() },
    }));
  }, []);

  const unassignCluster = useCallback((clusterKey) => {
    setSelections(prev => {
      const next = {};
      for (const l of TABS) {
        next[l] = { clusters: new Set(prev[l].clusters), images: new Set(prev[l].images) };
        next[l].clusters.delete(clusterKey);
      }
      return next;
    });
  }, []);

  const unassignImage = useCallback((resultId) => {
    setSelections(prev => {
      const next = {};
      for (const l of TABS) {
        next[l] = { clusters: new Set(prev[l].clusters), images: new Set(prev[l].images) };
        next[l].images.delete(resultId);
      }
      return next;
    });
  }, []);

  // ── 6.5: canSubmit ───────────────────────────────────────────────────────────

  const canSubmit = displayName.trim() && LABELS.every(l =>
    selections[l].clusters.size + selections[l].images.size >= 1
  );

  // ── 6.6: Submit button text ──────────────────────────────────────────────────

  const totalGroups = TABS.reduce((sum, l) => sum + selections[l].clusters.size, 0);
  const totalImages = TABS.reduce((sum, l) => sum + selections[l].images.size, 0);

  // ── 6.7: Submit handler ──────────────────────────────────────────────────────

  const submit = async () => {
    setSubmitting(true);
    setError('');
    setJobLog('');
    try {
      // Convert Sets to arrays for IPC serialization
      const selectionsPayload = {};
      for (const l of TABS) {
        selectionsPayload[l] = {
          clusters: Array.from(selections[l].clusters),
          images: Array.from(selections[l].images),
        };
      }
      const result = await window.electronAPI.createPrototypeVersion({
        reviewDbPath: paths.reviewDbPath,
        parent_review_run_id: parentReviewRunId,
        display_name: displayName.trim(),
        selections: selectionsPayload,
        set_active: setActive
      });
      if (result.error) { setError(result.error); setSubmitting(false); return; }
      if (result.selection_json_path) setJobLog(`selection_json=${result.selection_json_path}\n`);
      setJobSessionId(result.sessionId);
    } catch (e) {
      setError(String(e.message || e));
      setSubmitting(false);
    }
  };

  const updateFilters = useCallback((patch) => {
    setFilters((prev) => ({ ...prev, ...patch }));
  }, []);

  const clearFilters = useCallback(() => setFilters(defaultFilters()), []);

  const candidateResultIds = useCallback((candidate) => {
    const ids = new Set();
    for (const thumb of candidate.thumbnails || []) ids.add(Number(thumb.result_id));
    for (const row of assignmentsByKey[candidate.cluster_key] || []) ids.add(Number(row.result_id));
    return ids;
  }, [assignmentsByKey]);

  const hasClusterPick = useCallback((candidate, label) => selections[label].clusters.has(candidate.cluster_key), [selections]);

  const hasImagePick = useCallback((candidate, label) => {
    const ids = candidateResultIds(candidate);
    for (const id of ids) {
      if (selections[label].images.has(id)) return true;
    }
    return false;
  }, [candidateResultIds, selections]);

  const isPickedFor = useCallback((candidate, label) => hasClusterPick(candidate, label) || hasImagePick(candidate, label), [hasClusterPick, hasImagePick]);

  const isPickedAny = useCallback((candidate) => TABS.some((label) => isPickedFor(candidate, label)), [isPickedFor]);

  const matchesSearch = useCallback((candidate) => {
    const query = filters.search.trim().toLowerCase();
    if (!query) return true;
    if (String(candidate.cluster_key || '').toLowerCase().includes(query)) return true;
    for (const thumb of candidate.thumbnails || []) {
      if (String(thumb.result_id || '').toLowerCase().includes(query)) return true;
      if (String(thumb.image_path || '').toLowerCase().includes(query)) return true;
    }
    for (const row of assignmentsByKey[candidate.cluster_key] || []) {
      if (String(row.result_id || '').toLowerCase().includes(query)) return true;
      if (String(row.image_path || '').toLowerCase().includes(query)) return true;
    }
    return false;
  }, [assignmentsByKey, filters.search]);

  const bucketCounts = useMemo(() => {
    const counts = {};
    for (const candidate of candidates) counts[candidate.review_bucket] = (counts[candidate.review_bucket] || 0) + 1;
    return counts;
  }, [candidates]);

  const labelCounts = useMemo(() => {
    const counts = {};
    for (const candidate of candidates) counts[candidate.current_label || candidate.recommended_label] = (counts[candidate.current_label || candidate.recommended_label] || 0) + 1;
    return counts;
  }, [candidates]);

  const pickedCount = useMemo(() => candidates.filter((candidate) => isPickedAny(candidate)).length, [candidates, isPickedAny]);
  const excludedCount = useMemo(() => candidates.filter((candidate) => isPickedFor(candidate, 'excluded')).length, [candidates, isPickedFor]);

  const filteredCandidates = useMemo(() => {
    const out = candidates.filter((candidate) => {
      const label = candidate.current_label || candidate.recommended_label;
      if (!filters.buckets.includes(candidate.review_bucket)) return false;
      if (!filters.labels.includes(label)) return false;
      if (Number(candidate.top_score || 0) < filters.minTopScore) return false;
      if (Number(candidate.confidence_gap || 0) < filters.minConfidenceGap) return false;
      if (!matchesSearch(candidate)) return false;
      if (filters.picks === 'picked_this' && !isPickedFor(candidate, activeLabel)) return false;
      if (filters.picks === 'picked_any' && !isPickedAny(candidate)) return false;
      if (filters.picks === 'not_picked' && isPickedAny(candidate)) return false;
      if (filters.picks === 'excluded' && !isPickedFor(candidate, 'excluded')) return false;
      return true;
    });

    return [...out].sort((a, b) => {
      if (sort === 'cluster_key') return String(a.cluster_key).localeCompare(String(b.cluster_key));
      if (sort === 'recently_picked') {
        const recentA = Math.max(pickedAt.clusters[a.cluster_key] || 0, ...Array.from(candidateResultIds(a)).map((id) => pickedAt.images[id] || 0));
        const recentB = Math.max(pickedAt.clusters[b.cluster_key] || 0, ...Array.from(candidateResultIds(b)).map((id) => pickedAt.images[id] || 0));
        return recentB - recentA;
      }
      return Number(b[sort] || 0) - Number(a[sort] || 0);
    });
  }, [activeLabel, candidateResultIds, candidates, filters, isPickedAny, isPickedFor, matchesSearch, pickedAt, sort]);

  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title="New prototype version"
        left={<IconButton label="Back" onClick={onBack}><IconChevronLeft size={16} /></IconButton>}
        right={
          <Button variant="primary" disabled={!canSubmit || submitting} onClick={submit}>
            {submitting ? 'Running…' : `Run with ${totalGroups}g + ${totalImages}i`}
          </Button>
        }
      />
      {error && <div className="px-6 py-2"><ErrorMessage error={error} /></div>}

      {/* Config bar — 6.3: removed targetPerLabel input */}
      <div className="flex items-center gap-4 border-b border-[var(--border-muted)] px-6 py-3">
        <TextInput value={displayName} onChange={(e) => setDisplayName(e.currentTarget.value)} placeholder="Version name" className="w-[200px]" />
        <label className="flex items-center gap-1.5 text-[12px] text-[var(--text-muted)]">
          <input type="checkbox" checked={setActive} onChange={(e) => setSetActive(e.target.checked)} className="accent-[var(--primary)]" />
          Set as active
        </label>
      </div>

      <FilterBar
        activeLabel={activeLabel}
        tabs={TABS}
        selections={selections}
        filters={filters}
        sort={sort}
        sortOptions={SORT_OPTIONS}
        bucketOptions={ALL_REVIEW_BUCKETS.map((value) => ({ value, label: value }))}
        labelOptions={LABEL_FILTERS.map((value) => ({ value, label: value }))}
        bucketCounts={bucketCounts}
        labelCounts={labelCounts}
        totalCount={candidates.length}
        visibleCount={filteredCandidates.length}
        pickedCount={pickedCount}
        excludedCount={excludedCount}
        onActiveLabelChange={setActiveLabel}
        onFiltersChange={updateFilters}
        onSortChange={setSort}
        onClear={clearFilters}
      />

      {showLoaderStats && (
        <div className="border-b border-[var(--border-muted)] bg-[var(--surface-2)] px-6 py-1 text-[11px] text-[var(--text-muted)]">
          loader queued {loaderStats.queued} · inflight {loaderStats.inflight} · hit {loaderStats.cacheHit} · miss {loaderStats.cacheMiss} · memory {loaderStats.memoryMB}MB
        </div>
      )}

      {/* Job log */}
      {jobSessionId && (
        <pre className="max-h-[120px] overflow-auto border-b border-[var(--border-muted)] bg-[var(--surface-2)] px-6 py-2 text-[11px] text-[var(--text-muted)]">
          {jobLog || 'Starting…'}
        </pre>
      )}

      {/* Gallery */}
      <main className="min-h-0 flex-1 overflow-auto">
        {loading && <EmptyState title="Loading candidates" />}
        {!loading && filteredCandidates.length === 0 && (
          <EmptyState title="No candidates match filters">
            <Button onClick={clearFilters}>Clear filters</Button>
          </EmptyState>
        )}
        {!loading && filteredCandidates.map((c) => (
          <CandidateGalleryRow
            key={c.cluster_key}
            score={c}
            assignments={assignmentsByKey[c.cluster_key] || null}
            loadingRow={loadingKeys.has(c.cluster_key)}
            picked={selections[activeLabel].clusters.has(c.cluster_key)}
            excluded={selections.excluded.clusters.has(c.cluster_key)}
            activeLabel={activeLabel}
            selections={selections}
            onAssignCluster={assignCluster}
            onUnassignCluster={unassignCluster}
            onAssignImage={assignImage}
            onUnassignImage={unassignImage}
            onRequestAssignments={requestAssignments}
          />
        ))}
      </main>
    </div>
  );
}
