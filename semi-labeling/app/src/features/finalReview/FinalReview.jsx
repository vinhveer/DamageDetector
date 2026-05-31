import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconDatabase, IconDownload, IconFolderOpen, IconLoader2, IconPackageExport, IconRefresh } from '@tabler/icons-react';
import { Button, EmptyState, ErrorMessage, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';
import { getBitmap } from '../../utils/imageCache.js';

const VIEW_MODES = [
  { value: 'before', label: 'Before' },
  { value: 'after',  label: 'After'  },
  { value: 'diff',   label: 'Diff'   },
];

const SORT_OPTIONS = [
  { value: 'changed_desc', label: 'Most changed first' },
  { value: 'total_desc',   label: 'Most boxes first'   },
  { value: 'rel_path',     label: 'Path A→Z'           },
];

const LABEL_COLOR = {
  crack:  '#fbbf24',
  mold:   '#10b981',
  spall:  '#60a5fa',
  reject: '#f472b6',
};

const fmt = new Intl.NumberFormat('en-US');
const formatNum = (n) => fmt.format(Number(n || 0));

function PathField({ label, value, onChange, onBrowse, mode = 'file' }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-[12px] font-medium text-[var(--text-muted)]">{label}</span>
      <div className="flex gap-2">
        <TextInput value={value} onChange={(e) => onChange(e.currentTarget.value)} className="flex-1" />
        {onBrowse && (
          <Button onClick={() => onBrowse(mode)} className="shrink-0">
            <IconFolderOpen size={14} /> Browse
          </Button>
        )}
      </div>
    </label>
  );
}

function ConnectPanel({ paths, csvs, loading, error, onPathChange, onBrowse, onLoad, onPickFinalCsv }) {
  return (
    <div className="mx-auto grid w-full max-w-[880px] gap-4 p-6">
      <div className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)]">
        <div className="border-b border-[var(--border-muted)] px-4 py-3 text-[13px] font-medium text-[var(--text)]">
          Step 8 — Final review
        </div>
        <div className="grid gap-4 p-4">
          <PathField
            label="complete_labels CSV (Step 6 output)"
            value={paths.completeLabelsCsv}
            onChange={(v) => onPathChange('completeLabelsCsv', v)}
            onBrowse={(mode) => onBrowse('completeLabelsCsv', mode)}
          />
          <PathField
            label="final_labels CSV (Step 7 output)"
            value={paths.finalLabelsCsv}
            onChange={(v) => onPathChange('finalLabelsCsv', v)}
            onBrowse={(mode) => onBrowse('finalLabelsCsv', mode)}
          />
          {csvs.length > 1 && (
            <div className="rounded-[6px] border border-[var(--border-muted)] bg-[var(--surface-2)] p-2">
              <div className="mb-1 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">
                {csvs.length} final_labels CSVs found · pick one
              </div>
              <div className="grid max-h-[140px] gap-1 overflow-auto">
                {csvs.map((csv) => (
                  <button
                    key={csv.path}
                    type="button"
                    onClick={() => onPickFinalCsv(csv.path)}
                    className={cn(
                      'flex items-center justify-between rounded px-2 py-1 text-left text-[11px] font-mono',
                      paths.finalLabelsCsv === csv.path
                        ? 'bg-[var(--primary)]/15 text-[var(--text)]'
                        : 'text-[var(--text-muted)] hover:bg-[var(--hover)]'
                    )}
                  >
                    <span className="truncate">{csv.name}</span>
                    <span className="ml-2 shrink-0 text-[10px] text-[var(--text-subtle)]">
                      {new Date(csv.mtime_ms).toLocaleString()}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
          <PathField
            label="damage_scan.sqlite3 (Step 2 — for bbox coords)"
            value={paths.sourceDbPath}
            onChange={(v) => onPathChange('sourceDbPath', v)}
            onBrowse={(mode) => onBrowse('sourceDbPath', mode)}
          />
          <PathField
            label="Image root"
            value={paths.imageRootPath}
            onChange={(v) => onPathChange('imageRootPath', v)}
            onBrowse={(mode) => onBrowse('imageRootPath', mode)}
            mode="directory"
          />
          {error && <ErrorMessage message={error} />}
          <div className="flex justify-end">
            <Button
              variant="primary"
              onClick={onLoad}
              disabled={loading || !paths.completeLabelsCsv || !paths.finalLabelsCsv}
            >
              <IconDatabase size={14} /> Load images
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatsBar({
  totals, mode, onModeChange, sortBy, onSortChange,
  changedOnly, onChangedOnlyChange, onReload,
  onExportCoco, exporting,
}) {
  return (
    <div className="flex items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2 text-[12px]">
      <div className="text-[var(--text-muted)]">
        <span className="font-mono text-[var(--text)]">{formatNum(totals?.images)}</span> images ·
        <span className="ml-1 font-mono text-[var(--text)]">{formatNum(totals?.total)}</span> boxes ·
        <span className="ml-1 font-mono text-[#fbbf24]">{formatNum(totals?.changed)}</span> changed ·
        <span className="ml-1 font-mono text-[var(--text-muted)]">{formatNum(totals?.kept)}</span> kept
      </div>
      <div className="mx-2 h-4 w-px bg-[var(--border)]" />
      <div className="flex items-center gap-0.5 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] p-0.5">
        {VIEW_MODES.map((m) => (
          <button
            key={m.value}
            type="button"
            onClick={() => onModeChange(m.value)}
            className={cn(
              'h-6 rounded-[3px] px-2.5 text-[11px] font-medium transition-colors',
              mode === m.value
                ? 'bg-[var(--primary)] text-white'
                : 'text-[var(--text-muted)] hover:text-[var(--text)]'
            )}
          >
            {m.label}
          </button>
        ))}
      </div>
      <label className="flex items-center gap-1 text-[11px] text-[var(--text-muted)]">
        <input
          type="checkbox"
          checked={changedOnly}
          onChange={(e) => onChangedOnlyChange(e.target.checked)}
          className="h-3 w-3 accent-[var(--primary)]"
        />
        Changed only
      </label>
      <select
        value={sortBy}
        onChange={(e) => onSortChange(e.target.value)}
        className="h-7 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[11px]"
      >
        {SORT_OPTIONS.map((s) => (<option key={s.value} value={s.value}>{s.label}</option>))}
      </select>
      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={() => onExportCoco(false)}
          disabled={exporting}
          className="flex h-7 items-center gap-1 rounded-[5px] border border-[var(--primary)] bg-[var(--primary)]/15 px-2 text-[11px] font-medium text-[var(--primary)] hover:bg-[var(--primary)]/25 disabled:opacity-50"
          title="Export annotations.json only"
        >
          {exporting ? <IconLoader2 size={12} className="animate-spin" /> : <IconDownload size={12} />}
          Export JSON
        </button>
        <button
          type="button"
          onClick={() => onExportCoco(true)}
          disabled={exporting}
          className="flex h-7 items-center gap-1 rounded-[5px] border border-[var(--success)] bg-[var(--success)]/15 px-2 text-[11px] font-medium text-[var(--success)] hover:bg-[var(--success)]/25 disabled:opacity-50"
          title="Export annotations.json + copy images alongside"
        >
          {exporting ? <IconLoader2 size={12} className="animate-spin" /> : <IconPackageExport size={12} />}
          Export + Images
        </button>
        <button
          type="button"
          onClick={onReload}
          className="flex h-7 items-center gap-1 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2 text-[11px] text-[var(--text-muted)] hover:text-[var(--text)]"
        >
          <IconRefresh size={12} /> Reload
        </button>
      </div>
    </div>
  );
}

function ExportStatusDialog({ exporting, result, onDismiss }) {
  if (!exporting && !result) return null;
  return (
    <div
      role="status"
      className="fixed bottom-4 right-4 z-50 w-[380px] max-w-[calc(100vw-32px)] rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3 text-[12px] shadow-[0_2px_8px_rgba(0,0,0,0.4)]"
    >
      <div className="flex items-start gap-3">
        {exporting
          ? <IconLoader2 size={16} className="mt-0.5 shrink-0 animate-spin text-[var(--primary)]" />
          : <IconPackageExport size={16} className="mt-0.5 shrink-0 text-[var(--success)]" />}
        <div className="min-w-0 flex-1">
          <div className="font-medium text-[var(--text)]">
            {exporting ? 'Exporting COCO dataset' : 'COCO export complete'}
          </div>
          {result ? (
            <>
              <div className="mt-1 text-[var(--text-muted)]">
                {result.n_images} images, {result.n_annotations} annotations
                {result.n_copied > 0 && <span>, copied {result.n_copied} images</span>}
                {result.n_copy_errors > 0 && <span className="text-[var(--danger)]">, {result.n_copy_errors} copy errors</span>}
                {result.n_skipped_no_bbox > 0 && <span>, skipped {result.n_skipped_no_bbox} without bbox</span>}
              </div>
              <div className="mt-1 truncate font-mono text-[11px] text-[var(--text-muted)]">
                {result.annotations_path}
              </div>
            </>
          ) : (
            <div className="mt-1 text-[var(--text-muted)]">Choose an output folder and keep this window open.</div>
          )}
        </div>
        {!exporting && (
          <button
            type="button"
            onClick={onDismiss}
            className="shrink-0 rounded-[4px] px-1.5 py-0.5 text-[11px] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]"
          >
            Close
          </button>
        )}
      </div>
    </div>
  );
}

function ImageRow({ image, active, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'block w-full border-b border-[var(--border-muted)] px-3 py-2.5 text-left hover:bg-[var(--hover)]',
        active && 'bg-[var(--active)]'
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-medium text-[var(--text)]">{image.image_rel_path}</span>
        {image.changed > 0 && (
          <span className="shrink-0 rounded bg-[#fbbf24]/20 px-1.5 text-[10px] font-mono text-[#fbbf24]">
            {image.changed} Δ
          </span>
        )}
      </div>
      <div className="mt-0.5 text-[10px] font-mono text-[var(--text-muted)]">
        {image.total} boxes
      </div>
      <div className="mt-0.5 flex flex-wrap gap-1 text-[10px] font-mono">
        {['crack', 'mold', 'spall', 'reject'].map((cls) => {
          const before = image.counts_before?.[cls] || 0;
          const after = image.counts_after?.[cls] || 0;
          if (before === 0 && after === 0) return null;
          const changed = before !== after;
          return (
            <span
              key={cls}
              className="rounded px-1"
              style={{
                background: `${LABEL_COLOR[cls]}25`,
                color: LABEL_COLOR[cls],
              }}
            >
              {cls}: {before}{changed && `→${after}`}
            </span>
          );
        })}
      </div>
    </button>
  );
}

function CanvasView({ image, boxes, mode }) {
  const viewportRef = useRef(null);
  const canvasRef = useRef(null);
  const frameRef = useRef(null);
  const [source, setSource] = useState(null);
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });
  const [status, setStatus] = useState('idle');

  useEffect(() => {
    const node = viewportRef.current;
    if (!node || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setViewportSize({
        width: Math.floor(entry.contentRect.width),
        height: Math.floor(entry.contentRect.height),
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    setSource(null);
    setNaturalSize({ width: 0, height: 0 });
    if (!image?.image_uri) {
      setStatus('no-image');
      return undefined;
    }
    setStatus('loading');
    getBitmap(image.image_uri)
      .then((bitmap) => {
        if (cancelled) return;
        const w = Number(bitmap.naturalWidth || bitmap.width || 0);
        const h = Number(bitmap.naturalHeight || bitmap.height || 0);
        if (w <= 0 || h <= 0) { setStatus('error'); return; }
        setSource(bitmap);
        setNaturalSize({ width: w, height: h });
        setStatus('ready');
      })
      .catch(() => { if (!cancelled) setStatus('error'); });
    return () => { cancelled = true; };
  }, [image?.image_uri]);

  const displaySize = useMemo(() => {
    if (naturalSize.width <= 0 || naturalSize.height <= 0) return { width: 0, height: 0 };
    const maxW = Math.max(320, viewportSize.width - 24);
    const maxH = Math.max(260, viewportSize.height - 24);
    const scale = Math.min(maxW / naturalSize.width, maxH / naturalSize.height, 1);
    return {
      width: Math.max(1, Math.round(naturalSize.width * scale)),
      height: Math.max(1, Math.round(naturalSize.height * scale)),
    };
  }, [naturalSize.width, naturalSize.height, viewportSize.width, viewportSize.height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !source || displaySize.width <= 0 || displaySize.height <= 0) return undefined;
    if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
    frameRef.current = window.requestAnimationFrame(() => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(displaySize.width * dpr));
      canvas.height = Math.max(1, Math.round(displaySize.height * dpr));
      canvas.style.width = `${displaySize.width}px`;
      canvas.style.height = `${displaySize.height}px`;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, displaySize.width, displaySize.height);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(source, 0, 0, displaySize.width, displaySize.height);
    });
    return () => { if (frameRef.current) window.cancelAnimationFrame(frameRef.current); };
  }, [source, displaySize.width, displaySize.height]);

  if (!image) return <EmptyState title="Select an image" />;

  // Decide which label to show per box, and whether to highlight diff.
  // After mode hides reject (final dataset only keeps crack/mold/spall).
  const renderBoxes = boxes
    .filter((b) => {
      if (mode === 'after') return b.final_label !== 'reject';
      return true;
    })
    .map((b) => {
      let label;
      let color;
      let dashed = false;
      let bold = false;
      if (mode === 'before') {
        label = b.original_label;
        color = LABEL_COLOR[label] || '#999';
      } else if (mode === 'after') {
        label = b.final_label;
        color = LABEL_COLOR[label] || '#999';
      } else { // diff
        label = b.changed ? `${b.original_label}→${b.final_label}` : b.final_label;
        color = LABEL_COLOR[b.final_label] || '#999';
        dashed = !b.changed;
        bold = b.changed;
      }
      return { ...b, _label: label, _color: color, _dashed: dashed, _bold: bold };
    });

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-[var(--border-muted)] px-5 py-2.5">
        <div className="truncate text-[13px] font-medium text-[var(--text)]">{image.image_rel_path}</div>
        <div className="mt-0.5 text-[11px] text-[var(--text-muted)]">
          {mode === 'after'
            ? <>{renderBoxes.length} kept boxes <span className="text-[var(--text-subtle)]">/ {image.total} total · {image.counts_after?.reject || 0} dropped</span></>
            : <>{image.total} boxes · <span className="text-[#fbbf24]">{image.changed} changed</span></>
          }
          <span className="ml-3">
            {['crack', 'mold', 'spall', 'reject'].map((cls) => {
              const before = image.counts_before?.[cls] || 0;
              const after = image.counts_after?.[cls] || 0;
              if (before === 0 && after === 0) return null;
              if (mode === 'after' && cls === 'reject') return null;
              return (
                <span key={cls} className="mr-2 font-mono" style={{ color: LABEL_COLOR[cls] }}>
                  {cls} {mode === 'before' ? before : after}{mode === 'diff' && before !== after && ` (was ${before})`}
                </span>
              );
            })}
          </span>
        </div>
      </div>
      <div ref={viewportRef} className="min-h-0 flex-1 overflow-auto p-5">
        <div className="inline-block max-w-full rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3">
          <div className="relative inline-block max-w-full overflow-hidden rounded-[4px] bg-[var(--surface-2)] align-top">
            {status === 'ready' ? (
              <canvas ref={canvasRef} className="block" style={{ width: displaySize.width, height: displaySize.height }} />
            ) : (
              <div className="flex h-[360px] w-[520px] items-center justify-center text-[13px] text-[var(--text-muted)]">
                {status === 'error' ? 'Image error' : status === 'no-image' ? 'No image (check imageRootPath)' : 'Loading image'}
              </div>
            )}
            {status === 'ready' && naturalSize.width > 0 && (
              <svg
                className="pointer-events-none absolute left-0 top-0"
                width={displaySize.width}
                height={displaySize.height}
                viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
                preserveAspectRatio="none"
              >
                {renderBoxes.map((box) => {
                  const x1 = Math.max(0, Number(box.x1));
                  const y1 = Math.max(0, Number(box.y1));
                  const x2 = Math.max(x1, Number(box.x2));
                  const y2 = Math.max(y1, Number(box.y2));
                  return (
                    <g key={box.result_id}>
                      <rect
                        x={x1}
                        y={y1}
                        width={x2 - x1}
                        height={y2 - y1}
                        fill="transparent"
                        stroke={box._color}
                        strokeWidth={Math.max(box._bold ? 3 : 2, naturalSize.width / 700)}
                        strokeDasharray={box._dashed ? '6 4' : ''}
                        vectorEffect="non-scaling-stroke"
                      />
                      <text
                        x={x1 + 3}
                        y={Math.max(12, y1 + 13)}
                        fill={box._color}
                        fontSize="12"
                        fontWeight={box._bold ? '700' : '600'}
                      >
                        {box._label}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function FinalReview() {
  const [paths, setPaths] = useState({
    completeLabelsCsv: '',
    finalLabelsCsv: '',
    sourceDbPath: '',
    imageRootPath: '',
  });
  const [pathsLoaded, setPathsLoaded] = useState(false);
  const [csvs, setCsvs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [screen, setScreen] = useState('connect'); // connect | review
  const [images, setImages] = useState([]);
  const [totals, setTotals] = useState({ images: 0, total: 0, changed: 0, kept: 0 });
  const [selectedImagePath, setSelectedImagePath] = useState('');
  const [imageDetail, setImageDetail] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [detailLoading, setDetailLoading] = useState(false);

  const [mode, setMode] = useState('diff');
  const [changedOnly, setChangedOnly] = useState(true);
  const [sortBy, setSortBy] = useState('changed_desc');

  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState(null);

  // Load defaults
  useEffect(() => {
    if (pathsLoaded) return;
    let cancelled = false;
    Promise.all([
      window.electronAPI.getFinalReviewDefaults(),
      window.electronAPI.listFinalCsvs({ step7Dir: '' }).catch(() => ({ csvs: [] })),
    ])
      .then(([d, csvList]) => {
        if (cancelled) return;
        setPaths({
          completeLabelsCsv: d.completeLabelsCsv || '',
          finalLabelsCsv: d.finalLabelsCsv || '',
          sourceDbPath: d.sourceDbPath || '',
          imageRootPath: d.imageRootPath || '',
        });
        setCsvs(csvList.csvs || []);
      })
      .finally(() => { if (!cancelled) setPathsLoaded(true); });
    return () => { cancelled = true; };
  }, [pathsLoaded]);

  const handlePathChange = useCallback((field, value) => {
    setPaths((prev) => ({ ...prev, [field]: value }));
  }, []);

  const handleBrowse = useCallback(async (field, mode) => {
    try {
      const result = await window.electronAPI.browsePath(mode === 'directory' ? 'directory' : 'file');
      if (result) setPaths((prev) => ({ ...prev, [field]: result }));
    } catch (e) { setError(String(e?.message || e)); }
  }, []);

  const loadImages = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listFinalImages({
        completeLabelsCsv: paths.completeLabelsCsv,
        finalLabelsCsv: paths.finalLabelsCsv,
        changedOnly,
        sortBy,
      });
      setImages(result.images || []);
      setTotals(result.totals || { images: 0, total: 0, changed: 0, kept: 0 });
      setScreen('review');
      if ((result.images || []).length > 0) {
        setSelectedImagePath(result.images[0].image_rel_path);
      } else {
        setSelectedImagePath('');
      }
    } catch (e) {
      setError(String(e?.message || e));
    } finally { setLoading(false); }
  }, [paths.completeLabelsCsv, paths.finalLabelsCsv, changedOnly, sortBy]);

  // Reload images when filter/sort changes (only while in review screen)
  useEffect(() => {
    if (screen !== 'review') return;
    let cancelled = false;
    window.electronAPI.listFinalImages({
      completeLabelsCsv: paths.completeLabelsCsv,
      finalLabelsCsv: paths.finalLabelsCsv,
      changedOnly,
      sortBy,
    })
      .then((result) => {
        if (cancelled) return;
        setImages(result.images || []);
        setTotals(result.totals || { images: 0, total: 0, changed: 0, kept: 0 });
        const hasSelected = (result.images || []).some((i) => i.image_rel_path === selectedImagePath);
        if (!hasSelected && (result.images || []).length > 0) {
          setSelectedImagePath(result.images[0].image_rel_path);
        }
      })
      .catch((e) => { if (!cancelled) setError(String(e?.message || e)); });
    return () => { cancelled = true; };
  }, [screen, changedOnly, sortBy, paths.completeLabelsCsv, paths.finalLabelsCsv]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleExportCoco = useCallback(async (copyImages) => {
    setExporting(true);
    setExportResult(null);
    setError('');
    try {
      const outputDir = await window.electronAPI.browsePath('directory');
      if (!outputDir) {
        setExporting(false);
        return;
      }
      const res = await window.electronAPI.exportFinalToCoco({
        finalLabelsCsv: paths.finalLabelsCsv,
        sourceDbPath: paths.sourceDbPath,
        imageRootPath: paths.imageRootPath,
        outputDir,
        copyImages,
      });
      setExportResult(res);
    } catch (e) {
      setError(`Export failed: ${String(e?.message || e)}`);
    } finally {
      setExporting(false);
    }
  }, [paths.finalLabelsCsv, paths.sourceDbPath, paths.imageRootPath]);

  // Load detail when image changes
  useEffect(() => {
    if (screen !== 'review' || !selectedImagePath) {
      setImageDetail(null);
      setBoxes([]);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    window.electronAPI.getFinalImageBoxes({
      completeLabelsCsv: paths.completeLabelsCsv,
      finalLabelsCsv: paths.finalLabelsCsv,
      sourceDbPath: paths.sourceDbPath,
      imageRelPath: selectedImagePath,
      imageRootPath: paths.imageRootPath,
    })
      .then((res) => {
        if (cancelled) return;
        setImageDetail(res.image);
        setBoxes(res.boxes || []);
      })
      .catch((e) => { if (!cancelled) setError(String(e?.message || e)); })
      .finally(() => { if (!cancelled) setDetailLoading(false); });
    return () => { cancelled = true; };
  }, [screen, selectedImagePath, paths.completeLabelsCsv, paths.finalLabelsCsv, paths.sourceDbPath, paths.imageRootPath]);

  if (screen === 'connect') {
    return (
      <ConnectPanel
        paths={paths}
        csvs={csvs}
        loading={loading}
        error={error}
        onPathChange={handlePathChange}
        onBrowse={handleBrowse}
        onLoad={loadImages}
        onPickFinalCsv={(p) => setPaths((prev) => ({ ...prev, finalLabelsCsv: p }))}
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2">
        <button
          type="button"
          onClick={() => setScreen('connect')}
          className="h-7 rounded-[5px] border border-[var(--border)] px-2 text-[11px] text-[var(--text-muted)] hover:text-[var(--text)]"
        >
          ← Setup
        </button>
        <span className="truncate text-[12px] text-[var(--text-muted)]">
          {paths.finalLabelsCsv && (
            <>final: <span className="font-mono text-[var(--text)]">{paths.finalLabelsCsv.split('/').pop()}</span></>
          )}
        </span>
      </div>
      <StatsBar
        totals={totals}
        mode={mode}
        onModeChange={setMode}
        sortBy={sortBy}
        onSortChange={setSortBy}
        changedOnly={changedOnly}
        onChangedOnlyChange={setChangedOnly}
        onReload={loadImages}
        onExportCoco={handleExportCoco}
        exporting={exporting}
      />
      <ExportStatusDialog exporting={exporting} result={exportResult} onDismiss={() => setExportResult(null)} />
      {error && (
        <div className="border-b border-[var(--border-muted)] bg-[var(--danger)]/10 px-4 py-2 text-[12px] text-[var(--danger)]">
          {error}
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        <aside className="w-[300px] shrink-0 overflow-auto border-r border-[var(--border-muted)] bg-[var(--surface)]">
          {images.length === 0 && (
            <div className="px-3 py-4 text-[12px] text-[var(--text-muted)]">No images.</div>
          )}
          {images.map((img) => (
            <ImageRow
              key={img.image_rel_path}
              image={img}
              active={img.image_rel_path === selectedImagePath}
              onClick={() => setSelectedImagePath(img.image_rel_path)}
            />
          ))}
        </aside>
        <main className="min-w-0 flex-1">
          {detailLoading
            ? <div className="flex h-full items-center justify-center text-[13px] text-[var(--text-muted)]">Loading...</div>
            : <CanvasView image={imageDetail} boxes={boxes} mode={mode} />}
        </main>
      </div>
    </div>
  );
}
