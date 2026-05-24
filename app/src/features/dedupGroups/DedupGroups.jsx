import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconDatabase, IconFolderOpen, IconRefresh, IconSearch } from '@tabler/icons-react';
import { Button, EmptyState, ErrorMessage, SelectControl, TextInput } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';
import { getBitmap } from '../../utils/imageCache.js';

const LABELS = [
  { value: 'all', label: 'All labels' },
  { value: 'crack', label: 'crack' },
  { value: 'spall', label: 'spall' },
  { value: 'mold', label: 'mold' },
];

const VIEW_MODES = [
  { value: 'before', label: 'Before' },
  { value: 'after', label: 'After' },
];

const numberFormat = new Intl.NumberFormat('en-US');
const formatNumber = (value) => numberFormat.format(Number(value || 0));
const shortId = (value) => String(value || '').slice(0, 10);

function PageHeader({ title, subtitle, right }) {
  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-[var(--border-muted)] px-6">
      <div className="min-w-0">
        <span className="truncate text-[13px] font-medium text-[var(--text)]">{title}</span>
        {subtitle && <span className="ml-2 truncate text-[12px] text-[var(--text-muted)]">{subtitle}</span>}
      </div>
      {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
    </header>
  );
}

function PathField({ label, value, onChange, onBrowse, mode = 'file' }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-[12px] font-medium text-[var(--text-muted)]">{label}</span>
      <div className="flex gap-2">
        <TextInput value={value} onChange={(event) => onChange(event.currentTarget.value)} className="flex-1" />
        <Button onClick={() => onBrowse(mode)} className="shrink-0">
          <IconFolderOpen size={14} /> Browse
        </Button>
      </div>
    </label>
  );
}

function ConnectPanel({ paths, loading, onPathChange, onBrowse, onLoad }) {
  return (
    <div className="mx-auto grid w-full max-w-[880px] gap-4 p-6">
      <div className="rounded-[6px] border border-[var(--border)] bg-[var(--surface)]">
        <div className="border-b border-[var(--border-muted)] px-4 py-3 text-[13px] font-medium text-[var(--text)]">
          Step 4 database
        </div>
        <div className="grid gap-4 p-4">
          <PathField label="Dedup DB" value={paths.dedupDbPath} onChange={(value) => onPathChange('dedupDbPath', value)} onBrowse={(mode) => onBrowse('dedupDbPath', mode)} />
          <PathField label="Source DB" value={paths.sourceDbPath} onChange={(value) => onPathChange('sourceDbPath', value)} onBrowse={(mode) => onBrowse('sourceDbPath', mode)} />
          <PathField label="Image root" value={paths.imageRootPath} onChange={(value) => onPathChange('imageRootPath', value)} onBrowse={(mode) => onBrowse('imageRootPath', mode)} mode="directory" />
          <div className="flex justify-end">
            <Button variant="primary" onClick={onLoad} disabled={loading || !paths.dedupDbPath}>
              <IconDatabase size={14} /> Load images
            </Button>
          </div>
        </div>
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
        'block w-full border-b border-[var(--border-muted)] px-4 py-3 text-left hover:bg-[var(--hover)]',
        active && 'bg-[var(--active)]'
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <span className="truncate text-[13px] font-medium text-[var(--text)]">{image.image_rel_path}</span>
        <span className="text-[12px] tabular-nums text-[var(--text-muted)]">-{formatNumber(image.dropped_count)}</span>
      </div>
      <div className="mt-1 flex items-center gap-2 text-[11px] text-[var(--text-muted)]">
        <span>before {formatNumber(image.before_count)}</span>
        <span>after {formatNumber(image.after_count)}</span>
        <span>fused {formatNumber(image.fused_count)}</span>
      </div>
      <div className="mt-1 truncate text-[11px] text-[var(--text-subtle)]">
        crack {formatNumber(image.crack_count)} · spall {formatNumber(image.spall_count)} · mold {formatNumber(image.mold_count)}
      </div>
    </button>
  );
}

const boxStroke = (box) => {
  if (box.fused) return 'var(--primary)';
  if (!box.keep) return 'var(--danger)';
  if (box.predicted_label === 'crack') return 'var(--warning)';
  if (box.predicted_label === 'spall') return 'var(--success)';
  return 'var(--primary)';
};

function ImageBoxOverlay({ image, boxes, mode, loading }) {
  const viewportRef = useRef(null);
  const canvasRef = useRef(null);
  const frameRef = useRef(null);
  const [source, setSource] = useState(null);
  const [imageStatus, setImageStatus] = useState('idle');
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

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
      setImageStatus('no-image');
      return undefined;
    }

    setImageStatus('loading');
    getBitmap(image.image_uri)
      .then((bitmap) => {
        if (cancelled) return;
        const width = Number(bitmap.naturalWidth || bitmap.width || 0);
        const height = Number(bitmap.naturalHeight || bitmap.height || 0);
        if (width <= 0 || height <= 0) {
          setImageStatus('error');
          return;
        }
        setSource(bitmap);
        setNaturalSize({ width, height });
        setImageStatus('ready');
      })
      .catch(() => {
        if (!cancelled) setImageStatus('error');
      });

    return () => {
      cancelled = true;
      if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
    };
  }, [image?.image_uri]);

  const displaySize = useMemo(() => {
    if (naturalSize.width <= 0 || naturalSize.height <= 0) return { width: 0, height: 0 };
    const maxWidth = Math.max(320, viewportSize.width - 24);
    const maxHeight = Math.max(260, viewportSize.height - 24);
    const scale = Math.min(maxWidth / naturalSize.width, maxHeight / naturalSize.height, 1);
    return {
      width: Math.max(1, Math.round(naturalSize.width * scale)),
      height: Math.max(1, Math.round(naturalSize.height * scale)),
    };
  }, [naturalSize.height, naturalSize.width, viewportSize.height, viewportSize.width]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (loading || !canvas || !source || displaySize.width <= 0 || displaySize.height <= 0) return undefined;
    if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
    frameRef.current = window.requestAnimationFrame(() => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(displaySize.width * dpr));
      canvas.height = Math.max(1, Math.round(displaySize.height * dpr));
      canvas.style.width = `${displaySize.width}px`;
      canvas.style.height = `${displaySize.height}px`;
      const context = canvas.getContext('2d');
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, displaySize.width, displaySize.height);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      context.drawImage(source, 0, 0, displaySize.width, displaySize.height);
    });
    return () => {
      if (frameRef.current) window.cancelAnimationFrame(frameRef.current);
    };
  }, [displaySize.height, displaySize.width, loading, source]);

  if (!image) return <EmptyState title="Select an image" />;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-[var(--border-muted)] px-5 py-3">
        <div className="truncate text-[13px] font-medium text-[var(--text)]">{image.image_rel_path}</div>
        <div className="mt-1 text-[12px] text-[var(--text-muted)]">
          {mode === 'before' ? 'Before dedup' : 'After dedup'} · {formatNumber(boxes.length)} boxes
        </div>
      </div>
      <div ref={viewportRef} className="min-h-0 flex-1 overflow-auto p-5">
        {loading ? <EmptyState title="Loading boxes" /> : (
          <div className="inline-block max-w-full rounded-[6px] border border-[var(--border)] bg-[var(--surface)] p-3">
            <div className="relative inline-block max-w-full overflow-hidden rounded-[4px] bg-[var(--surface-2)] align-top">
              {imageStatus === 'ready' ? (
                <canvas ref={canvasRef} className="block" style={{ width: displaySize.width, height: displaySize.height }} />
              ) : (
                <div className="flex h-[360px] w-[520px] items-center justify-center text-[13px] text-[var(--text-muted)]">
                  {imageStatus === 'error' ? 'Image error' : imageStatus === 'no-image' ? 'No image' : 'Loading image'}
                </div>
              )}
              {imageStatus === 'ready' && naturalSize.width > 0 && naturalSize.height > 0 && (
                <svg
                  className="pointer-events-none absolute left-0 top-0"
                  width={displaySize.width}
                  height={displaySize.height}
                  viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
                  preserveAspectRatio="none"
                >
                  {boxes.map((box) => {
                    const x1 = Math.max(0, Number(box.x1));
                    const y1 = Math.max(0, Number(box.y1));
                    const x2 = Math.max(x1, Number(box.x2));
                    const y2 = Math.max(y1, Number(box.y2));
                    const stroke = boxStroke(box);
                    return (
                      <g key={`${box.result_id}-${box.fused ? 'f' : 'b'}`}>
                        <rect
                          x={x1}
                          y={y1}
                          width={x2 - x1}
                          height={y2 - y1}
                          fill="transparent"
                          stroke={stroke}
                          strokeWidth={Math.max(2, naturalSize.width / 700)}
                          vectorEffect="non-scaling-stroke"
                        />
                        <text x={x1 + 3} y={Math.max(12, y1 + 13)} fill={stroke} fontSize="12" fontWeight="600">
                          {box.predicted_label}{box.fused ? ' fused' : !box.keep ? ' drop' : ''}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function DedupGroups() {
  const [paths, setPaths] = useState({ dedupDbPath: '', sourceDbPath: '', imageRootPath: '' });
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [label, setLabel] = useState('all');
  const [mode, setMode] = useState('before');
  const [images, setImages] = useState([]);
  const [selectedImagePath, setSelectedImagePath] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [boxesLoading, setBoxesLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    window.electronAPI.getDedupGroupsDefaults()
      .then((defaults) => setPaths({
        dedupDbPath: defaults.dedupDbPath || '',
        sourceDbPath: defaults.sourceDbPath || '',
        imageRootPath: defaults.imageRootPath || '',
      }))
      .catch(() => {});
  }, []);

  const selectedRun = useMemo(
    () => runs.find((run) => run.dedup_run_id === selectedRunId) || null,
    [runs, selectedRunId]
  );

  const filteredImages = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return images;
    return images.filter((image) => String(image.image_rel_path || '').toLowerCase().includes(needle));
  }, [images, search]);

  const loadBoxes = useCallback(async (imageRelPath = selectedImagePath, nextMode = mode, nextLabel = label, runId = selectedRunId) => {
    if (!paths.dedupDbPath || !runId || !imageRelPath) return;
    setBoxesLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listDedupImageBoxes({
        dedupDbPath: paths.dedupDbPath,
        dedupRunId: runId,
        imageRelPath,
        mode: nextMode,
        label: nextLabel,
        sourceDbPath: paths.sourceDbPath,
        imageRootPath: paths.imageRootPath,
      });
      setSelectedImage(result.image || null);
      setBoxes(result.boxes || []);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBoxesLoading(false);
    }
  }, [label, mode, paths.dedupDbPath, paths.imageRootPath, paths.sourceDbPath, selectedImagePath, selectedRunId]);

  const loadImages = useCallback(async (runId = selectedRunId, nextLabel = label) => {
    if (!paths.dedupDbPath || !runId) return;
    setLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listDedupImages({
        dedupDbPath: paths.dedupDbPath,
        dedupRunId: runId,
        label: nextLabel,
        limit: 5000,
        sourceDbPath: paths.sourceDbPath,
        imageRootPath: paths.imageRootPath,
      });
      const nextImages = result.images || [];
      setImages(nextImages);
      const nextImagePath = nextImages[0]?.image_rel_path || '';
      setSelectedImagePath(nextImagePath);
      setSelectedImage(nextImages[0] || null);
      setBoxes([]);
      if (nextImagePath) await loadBoxes(nextImagePath, mode, nextLabel, runId);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  }, [label, loadBoxes, mode, paths.dedupDbPath, paths.imageRootPath, paths.sourceDbPath, selectedRunId]);

  const loadRuns = useCallback(async () => {
    if (!paths.dedupDbPath) return;
    setLoading(true);
    setError('');
    try {
      const result = await window.electronAPI.listDedupRuns({ dedupDbPath: paths.dedupDbPath });
      const nextRuns = result.runs || [];
      setRuns(nextRuns);
      const nextRunId = nextRuns[0]?.dedup_run_id || '';
      setSelectedRunId(nextRunId);
      if (nextRunId) await loadImages(nextRunId, label);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  }, [label, loadImages, paths.dedupDbPath]);

  const handleBrowse = async (key, browseMode) => {
    const nextPath = await window.electronAPI.browsePath(browseMode);
    if (nextPath) setPaths((prev) => ({ ...prev, [key]: nextPath }));
  };

  const handleRunChange = (runId) => {
    setSelectedRunId(runId);
    loadImages(runId, label);
  };

  const handleLabelChange = (nextLabel) => {
    setLabel(nextLabel);
    loadImages(selectedRunId, nextLabel);
  };

  const handleModeChange = (nextMode) => {
    setMode(nextMode);
    loadBoxes(selectedImagePath, nextMode, label);
  };

  const handleImageSelect = (image) => {
    setSelectedImagePath(image.image_rel_path);
    setSelectedImage(image);
    loadBoxes(image.image_rel_path, mode, label);
  };

  return (
    <div className="rv-enter flex h-full flex-col bg-[var(--bg)] rv-font">
      <PageHeader
        title="Dedup Groups"
        subtitle={selectedRun ? `${formatNumber(selectedRun.total_detections)} detections · ${shortId(selectedRun.dedup_run_id)}` : 'Step 4 before/after boxes'}
        right={runs.length > 0 && (
          <Button onClick={loadRuns} disabled={loading}>
            <IconRefresh size={14} /> Refresh
          </Button>
        )}
      />
      {error && <div className="px-6 py-2"><ErrorMessage error={error} /></div>}
      {runs.length === 0 ? (
        <main className="min-h-0 flex-1 overflow-auto">
          <ConnectPanel paths={paths} loading={loading} onPathChange={(key, value) => setPaths((prev) => ({ ...prev, [key]: value }))} onBrowse={handleBrowse} onLoad={loadRuns} />
        </main>
      ) : (
        <main className="grid min-h-0 flex-1 grid-cols-[360px_minmax(0,1fr)] overflow-hidden">
          <aside className="flex min-h-0 flex-col border-r border-[var(--border-muted)] bg-[var(--bg)]">
            <div className="grid gap-2 border-b border-[var(--border-muted)] p-3">
              <SelectControl value={selectedRunId} onChange={(event) => handleRunChange(event.currentTarget.value)}>
                {runs.map((run) => (
                  <option key={run.dedup_run_id} value={run.dedup_run_id}>{shortId(run.dedup_run_id)} · {formatNumber(run.total_detections)} detections</option>
                ))}
              </SelectControl>
              <div className="grid grid-cols-2 gap-2">
                <SelectControl value={label} onChange={(event) => handleLabelChange(event.currentTarget.value)}>
                  {LABELS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
                </SelectControl>
                <div className="grid grid-cols-2 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] p-0.5">
                  {VIEW_MODES.map((item) => (
                    <button
                      key={item.value}
                      type="button"
                      onClick={() => handleModeChange(item.value)}
                      className={cn(
                        'h-7 rounded-[4px] text-[12px] font-medium',
                        mode === item.value ? 'bg-[var(--active)] text-[var(--primary)]' : 'text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'
                      )}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex h-8 min-w-0 items-center gap-2 rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5">
                <IconSearch size={14} className="shrink-0 text-[var(--text-muted)]" />
                <input value={search} onChange={(event) => setSearch(event.currentTarget.value)} placeholder="Search image" className="min-w-0 flex-1 bg-transparent text-[13px] text-[var(--text)] outline-none placeholder:text-[var(--text-subtle)]" />
              </div>
            </div>
            <div className="min-h-0 flex-1 overflow-auto">
              {loading ? <EmptyState title="Loading images" /> : filteredImages.map((image) => (
                <ImageRow key={image.image_rel_path} image={image} active={image.image_rel_path === selectedImagePath} onClick={() => handleImageSelect(image)} />
              ))}
              {!loading && filteredImages.length === 0 && <EmptyState title="No images" />}
            </div>
          </aside>
          <section className="min-h-0 overflow-hidden">
            <ImageBoxOverlay image={selectedImage} boxes={boxes} mode={mode} loading={boxesLoading} />
          </section>
        </main>
      )}
    </div>
  );
}
