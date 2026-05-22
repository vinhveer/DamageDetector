import { useCallback, useEffect, useMemo } from 'react';
import {
  IconCirclePlus,
  IconCircleMinus,
  IconFolderOpen,
  IconPlayerPlay,
  IconTrash,
  IconPointer,
  IconTextCaption,
  IconArrowRight,
  IconCopy,
  IconCheck,
  IconAlertCircle,
  IconCircle,
  IconLoader,
  IconForbid,
} from '@tabler/icons-react';
import { useDispatch, useSelector } from 'react-redux';
import {
  goToStep,
  setCropActiveImage,
  setCropSegMode,
  addCropPoint,
  clearCropPoints,
  setCropTextPrompt,
  setCropPointMode,
  setCropSamCheckpoint,
  setCropGdinoCheckpoint,
  setCropDevice,
  setCropSkipped,
  runCropSegmentation,
} from '../inspectionWizardSlice.js';
import PointCanvas from '../../segment/components/PointCanvas.jsx';
import { Button, IconButton } from '../../../components/ui/index.js';
import { cn } from '../../../components/ui/cn.js';

const DEVICES = ['auto', 'cpu', 'cuda', 'mps'];

export default function Step1CropSetup() {
  const dispatch = useDispatch();
  const paths = useSelector((s) => s.inspectionWizard.source.paths);
  const crop = useSelector((s) => s.inspectionWizard.crop);
  const { activeImagePath, perImage, samCheckpoint, gdinoCheckpoint, device, boxThreshold, textThreshold } = crop;

  useEffect(() => {
    if (!activeImagePath && paths.length > 0) {
      dispatch(setCropActiveImage(paths[0]));
    }
  }, [activeImagePath, paths, dispatch]);

  const imgState = activeImagePath ? (perImage[activeImagePath] || null) : null;
  const segMode = imgState?.segMode ?? 'point';
  const points = imgState?.points ?? [];
  const pointMode = imgState?.pointMode ?? 'positive';
  const textPrompt = imgState?.textPrompt ?? '';
  const status = imgState?.status ?? 'idle';
  const skipped = Boolean(imgState?.skipped);
  const maskB64 = imgState?.maskB64 ?? null;
  const cropError = imgState?.cropError ?? null;

  const running = status === 'running';

  const doneCount = useMemo(() => {
    return paths.filter((p) => perImage[p]?.status === 'done').length;
  }, [paths, perImage]);
  const skippedCount = useMemo(() => {
    return paths.filter((p) => perImage[p]?.skipped).length;
  }, [paths, perImage]);

  const allReady = paths.length > 0 && paths.every((p) => perImage[p]?.status === 'done' || perImage[p]?.skipped);

  const browseSamCheckpoint = useCallback(async () => {
    const p = await window.electronAPI.browsePath('file');
    if (p) dispatch(setCropSamCheckpoint(p));
  }, [dispatch]);

  const browseGdinoCheckpoint = useCallback(async () => {
    const p = await window.electronAPI.browsePath('file');
    if (p) dispatch(setCropGdinoCheckpoint(p));
  }, [dispatch]);

  const handlePointAdded = useCallback(
    (pt) => {
      if (activeImagePath) dispatch(addCropPoint({ path: activeImagePath, point: pt }));
    },
    [dispatch, activeImagePath],
  );

  const canRun =
    !running &&
    !skipped &&
    activeImagePath &&
    samCheckpoint &&
    (segMode === 'text' ? textPrompt.trim().length > 0 : points.length > 0);

  const applyConfigToAll = useCallback(() => {
    if (!activeImagePath) return;
    const currentSegMode = perImage[activeImagePath]?.segMode ?? 'point';
    const currentTextPrompt = perImage[activeImagePath]?.textPrompt ?? '';
    paths.forEach((p) => {
      if (p !== activeImagePath) {
        dispatch(setCropSegMode({ path: p, segMode: currentSegMode }));
        dispatch(setCropTextPrompt({ path: p, textPrompt: currentTextPrompt }));
      }
    });
  }, [dispatch, activeImagePath, paths, perImage]);

  let canvasSrc = null;
  if (activeImagePath) {
    canvasSrc =
      status === 'done' && maskB64
        ? `data:image/png;base64,${maskB64}`
        : `file://${activeImagePath}`;
  }
  const canvasPoints = status === 'done' && maskB64 ? [] : points;

  const getStatusIcon = (imgStatus) => {
    switch (imgStatus) {
      case 'done': return <IconCheck size={11} stroke={3} className="text-[var(--success)]" />;
      case 'running': return <IconLoader size={11} className="animate-spin text-[var(--primary)]" />;
      case 'error': return <IconAlertCircle size={11} className="text-[var(--danger)]" />;
      case 'skipped': return <IconForbid size={11} className="text-[var(--warning)]" />;
      default: return <IconCircle size={11} className="text-[var(--text-muted)]" />;
    }
  };

  return (
    <div className="flex h-full min-h-0">
      <aside className="flex w-[220px] shrink-0 flex-col border-r border-[var(--border-muted)] bg-[var(--surface)]">
        <div className="border-b border-[var(--border-muted)] px-3 py-2.5">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">
            Images
          </span>
        </div>

        <div className="flex-1 overflow-y-auto p-1.5">
          <div className="flex flex-col gap-0.5">
            {paths.map((p) => {
              const imgStatus = perImage[p]?.skipped ? 'skipped' : (perImage[p]?.status ?? 'idle');
              const isActive = p === activeImagePath;
              return (
                <button
                  key={p}
                  type="button"
                  onClick={() => dispatch(setCropActiveImage(p))}
                  className={cn(
                    'flex items-center gap-2.5 rounded-[5px] px-2.5 py-1.5 text-left transition-colors',
                    isActive
                      ? 'bg-[var(--active)] text-[var(--text)]'
                      : 'text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
                  )}
                  title={p}
                >
                  <span className="shrink-0">{getStatusIcon(imgStatus)}</span>
                  <span className="truncate text-[12px]">{p.split('/').pop()}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="border-t border-[var(--border-muted)] p-3">
          <div className="mb-2 flex items-center justify-between text-[11px]">
            <span className="text-[var(--text-muted)]">Progress</span>
            <span className="font-medium text-[var(--text)]">
              <span className="text-[var(--success)]">{doneCount}</span>/{paths.length}
              {skippedCount ? <span className="text-[var(--warning)] ml-1">&middot; {skippedCount} skip</span> : ''}
            </span>
          </div>
          <div className="mb-3 h-1.5 w-full overflow-hidden rounded-full bg-[var(--surface-3)]">
            <div
              className="h-full rounded-full bg-[var(--success)] transition-all"
              style={{ width: `${paths.length > 0 ? (doneCount / paths.length) * 100 : 0}%` }}
            />
          </div>
          <Button
            variant="primary"
            onClick={() => dispatch(goToStep(2))}
            disabled={!allReady}
            className="flex w-full items-center justify-center gap-2"
          >
            Review & Export <IconArrowRight size={14} />
          </Button>
        </div>
      </aside>

      <div className="flex min-h-0 flex-1 flex-col">
        {/* === Toolbar === */}
        <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2.5">

          {/* ── Mode toggle ── */}
          <div className="flex rounded-[5px] border border-[var(--border)] overflow-hidden">
            <button
              type="button"
              onClick={() => activeImagePath && dispatch(setCropSegMode({ path: activeImagePath, segMode: 'point' }))}
              className={cn(
                'flex items-center gap-1.5 px-4 py-[7px] text-[13px] font-medium transition-colors',
                segMode === 'point'
                  ? 'bg-[var(--primary)] text-white'
                  : 'bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
              )}
            >
              <IconPointer size={14} /> Point
            </button>
            <button
              type="button"
              onClick={() => activeImagePath && dispatch(setCropSegMode({ path: activeImagePath, segMode: 'text' }))}
              className={cn(
                'flex items-center gap-1.5 px-4 py-[7px] text-[13px] font-medium transition-colors',
                segMode === 'text'
                  ? 'bg-[var(--primary)] text-white'
                  : 'bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
              )}
            >
              <IconTextCaption size={14} /> Text
            </button>
          </div>

          {/* ── Apply to all ── */}
          <button
            type="button"
            onClick={applyConfigToAll}
            className="flex items-center gap-1.5 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-3 py-[7px] text-[12px] font-medium text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)] transition-colors"
            title="Apply mode and text prompt to all images"
          >
            <IconCopy size={14} /> Apply to all
          </button>

          {/* ── Skip ── */}
          <button
            type="button"
            onClick={() => activeImagePath && dispatch(setCropSkipped({ path: activeImagePath, skipped: !skipped }))}
            className={cn(
              'flex items-center gap-1.5 rounded-[5px] border px-3 py-[7px] text-[12px] font-medium transition-colors',
              skipped
                ? 'border-[var(--warning)] bg-[var(--warning-bg)] text-[var(--warning)]'
                : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
            )}
          >
            <IconForbid size={14} /> {skipped ? 'Include' : 'Skip'}
          </button>

          <div className="h-5 w-px bg-[var(--border-muted)]" />

          {/* ── SAM checkpoint ── */}
          <div className="flex items-center gap-1.5">
            <span className="text-[12px] font-medium text-[var(--text-muted)]">SAM</span>
            <button
              type="button"
              onClick={browseSamCheckpoint}
              className="flex items-center gap-1.5 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-[7px] text-[12px] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)] transition-colors min-w-[60px] max-w-[140px] truncate"
              title={samCheckpoint || 'Click to select SAM checkpoint'}
            >
              {samCheckpoint ? samCheckpoint.split('/').pop() : 'None'}
            </button>
          </div>

          {/* ── DINO checkpoint ── */}
          {segMode === 'text' && (
            <div className="flex items-center gap-1.5">
              <span className="text-[12px] font-medium text-[var(--text-muted)]">DINO</span>
              <button
                type="button"
                onClick={browseGdinoCheckpoint}
                className="flex items-center gap-1.5 rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-[7px] text-[12px] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)] transition-colors min-w-[50px] max-w-[140px] truncate"
                title={gdinoCheckpoint || 'Click to select GDino checkpoint'}
              >
                {gdinoCheckpoint ? gdinoCheckpoint.split('/').pop() : 'Auto'}
              </button>
            </div>
          )}

          {/* ── Device ── */}
          <div className="flex items-center gap-1.5">
            <span className="text-[12px] font-medium text-[var(--text-muted)]">Device</span>
            <select
              value={device}
              onChange={(e) => dispatch(setCropDevice(e.target.value))}
              className="h-[30px] rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 text-[12px] text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
            >
              {DEVICES.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* ── Run ── */}
          <Button
            variant="primary"
            onClick={() => activeImagePath && dispatch(runCropSegmentation(activeImagePath))}
            disabled={!canRun}
            className="ml-auto flex items-center gap-2 h-[30px]"
          >
            {running ? (
              <>
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                Running…
              </>
            ) : (
              <>
                <IconPlayerPlay size={14} /> Run
              </>
            )}
          </Button>
        </div>

        {/* === Sub-controls === */}
        <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4 py-2">
          {segMode === 'point' && (
            <>
              <div className="flex rounded-[5px] border border-[var(--border)] overflow-hidden">
                <button
                  type="button"
                  onClick={() => activeImagePath && dispatch(setCropPointMode({ path: activeImagePath, pointMode: 'positive' }))}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-[7px] text-[12px] font-medium transition-colors',
                    pointMode === 'positive'
                      ? 'bg-[var(--success)] text-white'
                      : 'bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
                  )}
                >
                  <IconCirclePlus size={14} /> Positive
                </button>
                <button
                  type="button"
                  onClick={() => activeImagePath && dispatch(setCropPointMode({ path: activeImagePath, pointMode: 'negative' }))}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-[7px] text-[12px] font-medium transition-colors',
                    pointMode === 'negative'
                      ? 'bg-[var(--danger)] text-white'
                      : 'bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]',
                  )}
                >
                  <IconCircleMinus size={14} /> Negative
                </button>
              </div>

              {points.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-[12px] text-[var(--text-muted)]">
                    {points.length} point{points.length !== 1 ? 's' : ''}
                  </span>
                  <button
                    type="button"
                    onClick={() => activeImagePath && dispatch(clearCropPoints(activeImagePath))}
                    className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] hover:text-[var(--danger)] transition-colors"
                  >
                    <IconTrash size={13} /> Clear
                  </button>
                </div>
              )}
            </>
          )}

          {segMode === 'text' && (
            <div className="flex flex-1 items-center gap-3">
              <div className="flex min-w-[200px] flex-1 items-center gap-2">
                <span className="text-[12px] font-medium text-[var(--text-muted)]">Prompt</span>
                <input
                  type="text"
                  value={textPrompt}
                  onChange={(e) =>
                    activeImagePath && dispatch(setCropTextPrompt({ path: activeImagePath, textPrompt: e.target.value }))
                  }
                  placeholder="e.g. crack . peeling . mold"
                  className="flex-1 h-[30px] rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 text-[13px] text-[var(--text)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
                />
              </div>
              <span className="text-[12px] text-[var(--text-muted)]">
                Box {boxThreshold.toFixed(2)} &middot; Text {textThreshold.toFixed(2)}
              </span>
            </div>
          )}
        </div>

        {/* === Canvas === */}
        <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[var(--surface-2)]">
          {!activeImagePath ? (
            <div className="flex flex-col items-center gap-3 text-center text-[var(--text-muted)]">
              <div className="flex h-14 w-14 items-center justify-center rounded-full border border-[var(--border)] bg-[var(--surface)]">
                <IconFolderOpen size={28} strokeWidth={1.2} />
              </div>
              <p className="text-[13px]">Select an image from the sidebar</p>
            </div>
          ) : (
            <PointCanvas
              imageSrc={canvasSrc}
              points={canvasPoints}
              mode={pointMode}
              onPointAdded={segMode === 'point' ? handlePointAdded : undefined}
            />
          )}

          {activeImagePath && (
            <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5 rounded-[5px] border border-[var(--primary)] bg-[var(--bg)] px-2.5 py-1 text-[12px] font-medium text-[var(--primary)]">
              {segMode === 'point' ? <><IconPointer size={13} /> Point</> : <><IconTextCaption size={13} /> Text</>}
            </div>
          )}

          {running && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg)]/60">
              <div className="flex flex-col items-center gap-3 rounded-[6px] border border-[var(--border)] bg-[var(--surface)] px-6 py-4">
                <span className="h-7 w-7 animate-spin rounded-full border-2 border-[var(--primary)] border-t-transparent" />
                <span className="text-[12px] text-[var(--text-muted)]">
                  {segMode === 'text' ? 'Running GDino + SAM…' : 'Running SAM…'}
                </span>
              </div>
            </div>
          )}

          {status === 'error' && cropError && (
            <div className="absolute bottom-4 left-4 right-4 rounded-[5px] border border-[var(--danger)] bg-[var(--danger-bg)] p-3 text-[11px] text-[var(--danger)]">
              {cropError.split('\n')[0]}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
