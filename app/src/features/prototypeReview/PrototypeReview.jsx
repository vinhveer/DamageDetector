import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconChevronLeft, IconRefresh, IconEdit, IconX } from '@tabler/icons-react';
import { Button, EmptyState, ErrorMessage, IconButton, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';
import PageHeader from '../resultViewer/components/PageHeader.jsx';
import ImageGrid from '../resultViewer/components/ImageGrid.jsx';
import { formatFloat, formatNumber, groupRowsByImage, shortId } from '../resultViewer/utils.js';
import { imageLoaderPool } from '../../utils/imageLoaderPool.js';
import VersionsList from './VersionsList.jsx';
import VersionDetail from './VersionDetail.jsx';
import NewPrototypeSelector from './NewPrototypeSelector.jsx';

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

const scoreClass = (label, suggested) =>
  cn('tabular-nums', label === suggested ? 'font-semibold text-[var(--text)]' : 'text-[var(--text-muted)]');

const ASSIGNMENT_BATCH_DELAY_MS = 80;

const RELABEL_OPTIONS = ['crack', 'mold', 'spall'];


// ── Label context menu ───────────────────────────────────────────────────────

function LabelContextMenu({ menu, onSelect, onClose }) {
  useEffect(() => {
    if (!menu) return undefined;
    // Defer listener install so the very contextmenu event that opened the menu
    // doesn't immediately close it.
    const timer = window.setTimeout(() => {
      window.addEventListener('mousedown', onClose);
      window.addEventListener('contextmenu', onClose);
      window.addEventListener('scroll', onClose, true);
      window.addEventListener('keydown', onClose);
    }, 0);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('mousedown', onClose);
      window.removeEventListener('contextmenu', onClose);
      window.removeEventListener('scroll', onClose, true);
      window.removeEventListener('keydown', onClose);
    };
  }, [menu, onClose]);

  if (!menu) return null;
  const left = Math.min(menu.x, window.innerWidth - 160);
  const top = Math.min(menu.y, window.innerHeight - 140);
  const countLabel = menu.resultIds.length > 1 ? ` (${menu.resultIds.length})` : '';

  return (
    <div
      className="fixed z-50 min-w-[160px] rounded-[4px] border border-[var(--border)] bg-[var(--surface)] py-1 shadow-md"
      style={{ left, top }}
      onMouseDown={(e) => e.stopPropagation()}
      onContextMenu={(e) => { e.preventDefault(); e.stopPropagation(); }}
    >
      <div className="px-3 py-1 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
        Đổi label{countLabel}
      </div>
      {RELABEL_OPTIONS.map((label) => (
        <button
          key={label}
          type="button"
          onMouseDown={(e) => { e.stopPropagation(); onSelect(label); }}
          className="block w-full px-3 py-1.5 text-left text-[13px] text-[var(--text)] hover:bg-[var(--hover)]"
        >
          {label}
        </button>
      ))}
    </div>
  );
}


// ── CroppedThumb ──────────────────────────────────────────────────────────────

function CroppedThumb({ row, size = 120, priority = 1, selected, onToggle, onContextMenu }) {
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

  return (
    <button
      type="button"
      onClick={() => onToggle(Number(row.result_id))}
      onContextMenu={(e) => onContextMenu && onContextMenu(e, Number(row.result_id))}
      className={cn(
        'relative flex-shrink-0 cursor-pointer overflow-hidden rounded-[4px] border-2',
        selected ? 'border-[var(--primary)]' : 'border-[var(--border-muted)]'
      )}
      style={{ width: size, height: size }}
    >
      <canvas ref={canvasRef} width={size} height={size} className="block" />
      <span className="pointer-events-none absolute bottom-0 left-0 right-0 truncate bg-[rgba(0,0,0,0.55)] px-1 py-[1px] text-center text-[10px] font-medium uppercase tracking-wide text-white">
        {row.predicted_label || '—'}
      </span>
      {selected && (
        <div className="absolute right-1 top-1 flex h-[14px] w-[14px] items-center justify-center rounded-full bg-[var(--primary)]">
          <svg width="8" height="8" viewBox="0 0 8 8">
            <path d="M1 4L3 6.5L7 1.5" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      )}
    </button>
  );
}


// ── Batched edit loader ───────────────────────────────────────────────────────

function useBatchedAssignmentLoader(selectedRun, paths, enabled) {
  const [assignmentsByKey, setAssignmentsByKey] = useState({});
  const [loadingKeys, setLoadingKeys] = useState(new Set());
  const pendingKeysRef = useRef(new Set());
  const requestedKeysRef = useRef(new Set());
  const timerRef = useRef(null);

  const flush = useCallback(() => {
    timerRef.current = null;
    const clusterKeys = Array.from(pendingKeysRef.current);
    pendingKeysRef.current.clear();

    if (!enabled || !selectedRun || clusterKeys.length === 0) return;

    window.electronAPI.listPrototypeReviewAssignmentsBulk({
      reviewDbPath: paths.reviewDbPath,
      reviewRunId: selectedRun.review_run_id,
      featureDbPath: selectedRun.feature_db_path,
      sourceDbPath: selectedRun.source_db_path,
      imageRootPath: paths.imageRootPath,
      groupingRunId: selectedRun.grouping_run_id,
      clusterKeys
    })
      .then((result) => {
        const nextByKey = result.assignmentsByClusterKey || {};
        setAssignmentsByKey((prev) => {
          const next = { ...prev };
          for (const key of clusterKeys) next[key] = nextByKey[key] || [];
          return next;
        });
      })
      .catch(() => {
        setAssignmentsByKey((prev) => {
          const next = { ...prev };
          for (const key of clusterKeys) next[key] = [];
          return next;
        });
      })
      .finally(() => {
        setLoadingKeys((prev) => {
          const next = new Set(prev);
          for (const key of clusterKeys) next.delete(key);
          return next;
        });
      });
  }, [enabled, paths.imageRootPath, paths.reviewDbPath, selectedRun]);

  const requestAssignments = useCallback((clusterKey) => {
    if (!enabled || !selectedRun || !clusterKey || requestedKeysRef.current.has(clusterKey)) return;

    requestedKeysRef.current.add(clusterKey);
    pendingKeysRef.current.add(clusterKey);
    setLoadingKeys((prev) => {
      const next = new Set(prev);
      next.add(clusterKey);
      return next;
    });

    if (!timerRef.current) {
      timerRef.current = window.setTimeout(flush, ASSIGNMENT_BATCH_DELAY_MS);
    }
  }, [enabled, flush, selectedRun]);

  useEffect(() => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    pendingKeysRef.current.clear();
    requestedKeysRef.current.clear();
    setAssignmentsByKey({});
    setLoadingKeys(new Set());
  }, [
    paths.imageRootPath,
    paths.reviewDbPath,
    selectedRun?.feature_db_path,
    selectedRun?.grouping_run_id,
    selectedRun?.review_run_id,
    selectedRun?.source_db_path
  ]);

  useEffect(() => () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
  }, []);

  const patchAssignments = useCallback((resultIds, patch) => {
    const idSet = new Set(resultIds.map(Number));
    setAssignmentsByKey((prev) => {
      const next = {};
      for (const [key, list] of Object.entries(prev)) {
        next[key] = list.map((row) =>
          idSet.has(Number(row.result_id)) ? { ...row, ...patch } : row
        );
      }
      return next;
    });
  }, []);

  return { assignmentsByKey, loadingKeys, requestAssignments, patchAssignments };
}


// ── ClusterEditRow ────────────────────────────────────────────────────────────

function ClusterEditRow({ score, assignments, loadingRow, selectedIds, deletedIds, onToggleImage, onToggleGroup, onRequestAssignments, onContextMenuImage, onContextMenuGroup }) {
  const rowRef = useRef(null);
  const checkRef = useRef(null);
  const [thumbPriority, setThumbPriority] = useState(2);

  const visibleAssignments = useMemo(
    () => (assignments || []).filter((a) => !deletedIds.has(Number(a.result_id))),
    [assignments, deletedIds]
  );

  const resultIds = useMemo(() => visibleAssignments.map((a) => Number(a.result_id)), [visibleAssignments]);
  const groupSelected = resultIds.length > 0 && resultIds.every((id) => selectedIds.has(id));
  const anySelected = resultIds.some((id) => selectedIds.has(id));

  useEffect(() => {
    if (checkRef.current) checkRef.current.indeterminate = anySelected && !groupSelected;
  }, [anySelected, groupSelected]);

  // Load assignments lazily when the row scrolls into view
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

  return (
    <div ref={rowRef} className="border-b border-[var(--border-muted)] px-6 py-3">
      <div
        className="mb-2 flex items-center gap-3"
        onContextMenu={(e) => onContextMenuGroup && resultIds.length > 0 && onContextMenuGroup(e, resultIds)}
      >
        <input
          ref={checkRef}
          type="checkbox"
          checked={groupSelected}
          onChange={() => resultIds.length > 0 && onToggleGroup(resultIds)}
          className="h-4 w-4 cursor-pointer accent-[var(--primary)]"
        />
        <span className="font-semibold text-[var(--text)]">{score.cluster_key}</span>
        <span className={cn('rounded-[4px] border px-1.5 py-0.5 text-[11px] font-medium', bucketClass(score.review_bucket))}>
          {score.review_bucket}
        </span>
        <span className="text-[12px] text-[var(--text-muted)]">
          {score.original_major_label} → <span className="font-medium text-[var(--text)]">{score.recommended_label}</span>
          {' · '}top {formatFloat(score.top_score, 3)}
          {' · '}{formatNumber(score.cluster_size)} ảnh
        </span>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {loadingRow && (
          <div className="flex h-[120px] items-center text-[12px] text-[var(--text-muted)]">Loading…</div>
        )}
        {visibleAssignments.map((row) => (
          <CroppedThumb
            key={row.result_id}
              row={row}
              size={120}
              priority={thumbPriority}
              selected={selectedIds.has(Number(row.result_id))}
            onToggle={onToggleImage}
            onContextMenu={onContextMenuImage}
          />
        ))}
        {assignments !== null && visibleAssignments.length === 0 && !loadingRow && (
          <div className="text-[12px] text-[var(--text-muted)]">No images</div>
        )}
      </div>
    </div>
  );
}


// ── Main component ────────────────────────────────────────────────────────────

function PrototypeReviewLegacy() {
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

  // Edit mode state
  const [editMode, setEditMode] = useState(false);
  const [selectedResultIds, setSelectedResultIds] = useState(new Set());
  const [deletedIds, setDeletedIds] = useState(new Set());
  const [deleting, setDeleting] = useState(false);

  const selectedRun = runs.find((run) => run.review_run_id === selectedRunId) || null;
  const imageGroups = useMemo(() => groupRowsByImage(assignments), [assignments]);

  const visibleScores = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return scores;
    return scores.filter((item) =>
      [item.cluster_key, item.original_major_label, item.recommended_label, item.review_bucket, item.reason]
        .some((value) => String(value || '').toLowerCase().includes(needle))
    );
  }, [scores, query]);

  // Edit mode: sort by top_score DESC (highest similarity first)
  const editScores = useMemo(
    () => [...visibleScores].sort((a, b) => b.top_score - a.top_score),
    [visibleScores]
  );
  const {
    assignmentsByKey: editAssignmentsByKey,
    loadingKeys: editLoadingKeys,
    requestAssignments: requestEditAssignments,
    patchAssignments: patchEditAssignments
  } = useBatchedAssignmentLoader(selectedRun, paths, editMode);

  const [labelMenu, setLabelMenu] = useState(null); // {x, y, resultIds}

  const loadRuns = useCallback(async (nextPaths) => {
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
      setEditMode(false);
      setSelectedResultIds(new Set());
      setDeletedIds(new Set());
    } catch (event) {
      setError(String(event.message || event));
      setRuns([]);
      setSelectedRunId('');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadScores = useCallback(async () => {
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
      setEditMode(false);
      setSelectedResultIds(new Set());
      setDeletedIds(new Set());
    } catch (event) {
      setError(String(event.message || event));
      setScores([]);
    } finally {
      setLoading(false);
    }
  }, [bucket, label, paths.reviewDbPath, selectedRunId]);

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

  const toggleImage = useCallback((resultId) => {
    setSelectedResultIds((prev) => {
      const next = new Set(prev);
      if (next.has(resultId)) next.delete(resultId);
      else next.add(resultId);
      return next;
    });
  }, []);

  const toggleGroup = useCallback((resultIds) => {
    setSelectedResultIds((prev) => {
      const next = new Set(prev);
      const allIn = resultIds.every((id) => next.has(id));
      if (allIn) resultIds.forEach((id) => next.delete(id));
      else resultIds.forEach((id) => next.add(id));
      return next;
    });
  }, []);

  const openLabelMenu = useCallback((event, resultIds) => {
    event.preventDefault();
    event.stopPropagation();
    const ids = resultIds.map(Number).filter(Number.isFinite);
    if (!ids.length) return;
    setLabelMenu({ x: event.clientX, y: event.clientY, resultIds: ids });
  }, []);

  const openLabelMenuForImage = useCallback((event, resultId) => {
    openLabelMenu(event, [resultId]);
  }, [openLabelMenu]);

  const closeLabelMenu = useCallback(() => setLabelMenu(null), []);

  const applyLabel = useCallback(async (label) => {
    if (!labelMenu || !selectedRun) return;
    const { resultIds } = labelMenu;
    setLabelMenu(null);
    setError('');
    try {
      await window.electronAPI.setPrototypeAssignmentsLabel({
        featureDbPath: selectedRun.feature_db_path,
        groupingRunId: selectedRun.grouping_run_id,
        resultIds,
        label
      });
      patchEditAssignments(resultIds, { predicted_label: label });
    } catch (event) {
      setError(String(event.message || event));
    }
  }, [labelMenu, selectedRun, patchEditAssignments]);

  const deleteSelected = async () => {
    if (!selectedResultIds.size || !selectedRun) return;
    setDeleting(true);
    setError('');
    try {
      const ids = Array.from(selectedResultIds);
      await window.electronAPI.markPrototypeAssignmentsAsOutlier({
        featureDbPath: selectedRun.feature_db_path,
        groupingRunId: selectedRun.grouping_run_id,
        resultIds: ids
      });
      setDeletedIds((prev) => new Set([...prev, ...ids]));
      setSelectedResultIds(new Set());
    } catch (event) {
      setError(String(event.message || event));
    } finally {
      setDeleting(false);
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
  }, [loadRuns]);

  useEffect(() => {
    loadScores();
  }, [loadScores]);

  // ── Detail view ───────────────────────────────────────────────────────────
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

  // ── Main view ─────────────────────────────────────────────────────────────
  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title="Prototype Review"
        subtitle={selectedRun ? `${formatNumber(selectedRun.total_clusters)} groups · ${shortId(selectedRun.review_run_id)}` : 'Run Step 5 to generate prototype_review.sqlite3'}
        right={
          <>
            <SelectControl value={selectedRunId} onChange={(e) => setSelectedRunId(e.currentTarget.value)} className="w-[280px]" disabled={runs.length === 0}>
              <option value="">No review run</option>
              {runs.map((run) => (
                <option key={run.review_run_id} value={run.review_run_id}>
                  {run.created_at_utc?.slice(0, 16) || 'Unknown'} · {shortId(run.review_run_id)}
                </option>
              ))}
            </SelectControl>
            <Button onClick={() => loadRuns(paths)}>
              <IconRefresh size={14} />
              Reload
            </Button>
          </>
        }
      />

      {/* Filter bar */}
      <div className="flex min-h-14 items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--bg)] px-6 py-3">
        <SelectControl value={bucket} onChange={(e) => setBucket(e.currentTarget.value)} className="w-[150px]" disabled={editMode}>
          {BUCKETS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </SelectControl>
        <SelectControl value={label} onChange={(e) => setLabel(e.currentTarget.value)} className="w-[140px]" disabled={editMode}>
          {LABELS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
        </SelectControl>
        <TextInput value={query} onChange={(e) => setQuery(e.currentTarget.value)} placeholder="Filter groups" className="w-[280px]" disabled={editMode} />
        <div className="ml-auto flex items-center gap-3">
          <span className="text-[13px] text-[var(--text-muted)]">{formatNumber(visibleScores.length)} groups</span>
          {runs.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setEditMode((prev) => !prev);
                setSelectedResultIds(new Set());
              }}
              className={cn(
                'flex items-center gap-1.5 rounded-[4px] border px-3 py-1.5 text-[13px]',
                editMode
                  ? 'border-[var(--primary)] bg-[var(--primary-bg)] text-[var(--primary)]'
                  : 'border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)]'
              )}
            >
              <IconEdit size={14} />
              Edit
            </button>
          )}
        </div>
      </div>

      {/* Edit mode action bar */}
      {editMode && (
        <div className="flex items-center gap-4 border-b border-[var(--border-muted)] bg-[var(--surface)] px-6 py-2">
          <span className="text-[13px] text-[var(--text-muted)]">
            {selectedResultIds.size > 0 ? `${selectedResultIds.size} ảnh đã chọn` : 'Chọn ảnh để xóa'}
          </span>
          <button
            type="button"
            disabled={selectedResultIds.size === 0 || deleting}
            onClick={deleteSelected}
            className="rounded-[4px] border border-[var(--danger)] px-3 py-1 text-[13px] text-[var(--danger)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {deleting ? 'Đang xóa…' : `Xóa (${selectedResultIds.size})`}
          </button>
          {selectedResultIds.size > 0 && (
            <button
              type="button"
              onClick={() => setSelectedResultIds(new Set())}
              className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] hover:text-[var(--text)]"
            >
              <IconX size={12} />
              Bỏ chọn
            </button>
          )}
          {deletedIds.size > 0 && (
            <span className="ml-auto text-[12px] text-[var(--text-muted)]">
              {deletedIds.size} ảnh đã đánh dấu xóa trong phiên này
            </span>
          )}
        </div>
      )}

      {error && <div className="px-8 py-3"><ErrorMessage error={error} /></div>}

      {/* Content */}
      <main className="min-h-0 flex-1 overflow-auto bg-[var(--bg)]">
        {loading && <EmptyState title="Loading" />}
        {!loading && runs.length === 0 && <EmptyState title="No Step 5 review DB" />}

        {/* Edit mode: grouped thumbnail strips sorted by top_score DESC */}
        {!loading && runs.length > 0 && editMode && selectedRun && (
          <div>
            {editScores.length === 0 && <EmptyState title="No matching groups" />}
            {editScores.map((score) => (
              <ClusterEditRow
                key={score.cluster_key}
                score={score}
                assignments={editAssignmentsByKey[score.cluster_key] || null}
                loadingRow={editLoadingKeys.has(score.cluster_key)}
                selectedIds={selectedResultIds}
                deletedIds={deletedIds}
                onToggleImage={toggleImage}
                onToggleGroup={toggleGroup}
                onRequestAssignments={requestEditAssignments}
                onContextMenuImage={openLabelMenuForImage}
                onContextMenuGroup={openLabelMenu}
              />
            ))}
          </div>
        )}

        {/* Normal mode: table */}
        {!loading && runs.length > 0 && !editMode && (
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
                    <span className={cn('rounded-[4px] border px-1.5 py-0.5 text-[11px] font-medium', bucketClass(score.review_bucket))}>
                      {score.review_bucket}
                    </span>
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

        {!loading && runs.length > 0 && !editMode && visibleScores.length === 0 && (
          <EmptyState title="No matching groups" />
        )}
      </main>
      <LabelContextMenu menu={labelMenu} onSelect={applyLabel} onClose={closeLabelMenu} />
    </div>
  );
}


// ── View router ─────────────────────────────────────────────────────────────

export default function PrototypeReview() {
  const [view, setView] = useState('versions'); // 'versions' | 'detail' | 'new' | 'legacy'
  const [paths, setPaths] = useState({ reviewDbPath: '', imageRootPath: '' });
  const [activeRunId, setActiveRunId] = useState(null);
  const [parentRunId, setParentRunId] = useState(null);

  useEffect(() => {
    window.electronAPI.getPrototypeReviewDefaults().then((defaults) => {
      setPaths({ reviewDbPath: defaults.reviewDbPath || '', imageRootPath: defaults.imageRootPath || '' });
    }).catch(() => {});
  }, []);

  if (!paths.reviewDbPath) return <EmptyState title="Loading…" />;

  if (view === 'detail' && activeRunId) {
    return (
      <VersionDetail
        paths={paths}
        reviewRunId={activeRunId}
        onBack={() => setView('versions')}
        onNewVersion={(rid) => { setParentRunId(rid); setView('new'); }}
      />
    );
  }

  if (view === 'new' && parentRunId) {
    return (
      <NewPrototypeSelector
        paths={paths}
        parentReviewRunId={parentRunId}
        onBack={() => setView('versions')}
        onCreated={(newId) => { if (newId) { setActiveRunId(newId); setView('detail'); } else { setView('versions'); } }}
      />
    );
  }

  if (view === 'legacy') {
    return <PrototypeReviewLegacy />;
  }

  return (
    <VersionsList
      paths={paths}
      onOpenVersion={(rid) => { setActiveRunId(rid); setView('detail'); }}
      onNewVersion={(rid) => { setParentRunId(rid); setView('new'); }}
    />
  );
}
