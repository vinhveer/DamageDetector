import { useCallback } from 'react';
import {
  IconCirclePlus,
  IconCircleMinus,
  IconFolderOpen,
  IconPlayerPlay,
  IconTrash,
  IconToggleLeft,
  IconToggleRight,
  IconX,
  IconPointer,
  IconTextCaption,
} from '@tabler/icons-react';
import { useDispatch, useSelector } from 'react-redux';
import {
  addPoint,
  clearPoints,
  removePoint,
  runSegmentation,
  setDevice,
  setGdinoCheckpoint,
  setImagePath,
  setPointMode,
  setSamCheckpoint,
  setSegMode,
  setTextPrompt,
  setBoxThreshold,
  setTextThreshold,
  toggleOverlay,
} from './segmentSlice.js';
import PointCanvas from './components/PointCanvas.jsx';
import { Button, IconButton } from '../../components/ui/index.js';
import { cn } from '../../components/ui/cn.js';

const DEVICES = ['auto', 'cpu', 'cuda', 'mps'];

export default function SegmentTab() {
  const dispatch = useDispatch();
  const {
    imagePath,
    points,
    pointMode,
    segMode,
    textPrompt,
    gdinoCheckpoint,
    status,
    result,
    error,
    samCheckpoint,
    device,
    showOverlay,
    boxThreshold,
    textThreshold,
  } = useSelector((s) => s.segment);

  const running = status === 'running';

  const browseImage = useCallback(async () => {
    const p = await window.electronAPI.browsePath('file');
    if (p) dispatch(setImagePath(p));
  }, [dispatch]);

  const browseCheckpoint = useCallback(async () => {
    const p = await window.electronAPI.browsePath('file');
    if (p) dispatch(setSamCheckpoint(p));
  }, [dispatch]);

  const browseGdino = useCallback(async () => {
    const p = await window.electronAPI.browsePath('file');
    if (p) dispatch(setGdinoCheckpoint(p));
  }, [dispatch]);

  const handlePointAdded = useCallback((pt) => {
    dispatch(addPoint(pt));
  }, [dispatch]);

  const canRun = !running && imagePath && samCheckpoint && (
    segMode === 'text' ? textPrompt.trim().length > 0 : points.length > 0
  );

  let canvasSrc = null;
  if (imagePath) {
    canvasSrc = (showOverlay && result?.overlayB64)
      ? `data:image/png;base64,${result.overlayB64}`
      : `file://${imagePath}`;
  }

  const canvasPoints = (showOverlay && result?.overlayB64) ? [] : points;

  return (
    <div className="flex h-full flex-col bg-[var(--bg)] rv-font">
      <div className="flex min-h-0 flex-1 overflow-hidden">

        {/* ── Left controls panel ── */}
        <aside className="flex w-[272px] shrink-0 flex-col gap-4 overflow-y-auto border-r border-[var(--border-muted)] bg-[var(--surface)] p-4">

          {/* Segmentation mode toggle */}
          <section className="flex flex-col gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Mode</span>
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={() => dispatch(setSegMode('point'))}
                className={cn(
                  'flex flex-1 items-center justify-center gap-1.5 rounded-[5px] border py-1.5 text-[12px] font-medium',
                  segMode === 'point'
                    ? 'border-[var(--primary)] bg-[var(--primary-bg,#1e3a5f)] text-[var(--primary)]'
                    : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'
                )}
              >
                <IconPointer size={13} />
                Point
              </button>
              <button
                type="button"
                onClick={() => dispatch(setSegMode('text'))}
                className={cn(
                  'flex flex-1 items-center justify-center gap-1.5 rounded-[5px] border py-1.5 text-[12px] font-medium',
                  segMode === 'text'
                    ? 'border-[var(--primary)] bg-[var(--primary-bg,#1e3a5f)] text-[var(--primary)]'
                    : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'
                )}
              >
                <IconTextCaption size={13} />
                Text
              </button>
            </div>
          </section>

          {/* Image picker */}
          <section className="flex flex-col gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Image</span>
            <div className="flex gap-1.5">
              <div
                className="flex min-w-0 flex-1 items-center truncate rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[12px] text-[var(--text-muted)]"
                title={imagePath || ''}
              >
                <span className="truncate">{imagePath ? imagePath.split('/').pop() : 'No image selected'}</span>
              </div>
              <IconButton label="Browse image" onClick={browseImage}>
                <IconFolderOpen size={15} />
              </IconButton>
            </div>
          </section>

          {/* SAM checkpoint */}
          <section className="flex flex-col gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">SAM Checkpoint</span>
            <div className="flex gap-1.5">
              <div
                className="flex min-w-0 flex-1 items-center truncate rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[12px] text-[var(--text-muted)]"
                title={samCheckpoint || ''}
              >
                <span className="truncate">{samCheckpoint ? samCheckpoint.split('/').pop() : 'No checkpoint'}</span>
              </div>
              <IconButton label="Browse checkpoint" onClick={browseCheckpoint}>
                <IconFolderOpen size={15} />
              </IconButton>
            </div>
          </section>

          {/* ── Point mode controls ── */}
          {segMode === 'point' && (<>
            <section className="flex flex-col gap-1.5">
              <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Point Mode</span>
              <div className="flex gap-1.5">
                <button
                  type="button"
                  onClick={() => dispatch(setPointMode('positive'))}
                  className={cn(
                    'flex flex-1 items-center justify-center gap-1.5 rounded-[5px] border py-1.5 text-[12px] font-medium',
                    pointMode === 'positive'
                      ? 'border-[var(--success)] bg-[var(--success-bg)] text-[var(--success)]'
                      : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'
                  )}
                >
                  <IconCirclePlus size={14} />
                  Positive
                </button>
                <button
                  type="button"
                  onClick={() => dispatch(setPointMode('negative'))}
                  className={cn(
                    'flex flex-1 items-center justify-center gap-1.5 rounded-[5px] border py-1.5 text-[12px] font-medium',
                    pointMode === 'negative'
                      ? 'border-[var(--danger)] bg-[var(--danger-bg)] text-[var(--danger)]'
                      : 'border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:bg-[var(--hover)] hover:text-[var(--text)]'
                  )}
                >
                  <IconCircleMinus size={14} />
                  Negative
                </button>
              </div>
              <p className="text-[11px] text-[var(--text-muted)]">Click on the image to place points.</p>
            </section>

            {/* Points list */}
            <section className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
                  Points {points.length > 0 && `(${points.length})`}
                </span>
                {points.length > 0 && (
                  <button
                    type="button"
                    onClick={() => dispatch(clearPoints())}
                    className="flex items-center gap-1 text-[11px] text-[var(--text-muted)] hover:text-[var(--danger)]"
                  >
                    <IconTrash size={12} />
                    Clear all
                  </button>
                )}
              </div>
              {points.length === 0 ? (
                <div className="rounded-[5px] border border-[var(--border-muted)] bg-[var(--surface-2)] px-3 py-2.5 text-[12px] text-[var(--text-muted)]">
                  No points yet — click the image
                </div>
              ) : (
                <div className="flex max-h-[180px] flex-col gap-0.5 overflow-y-auto">
                  {points.map((p, i) => (
                    <div key={i} className="flex items-center gap-2 rounded-[4px] px-2 py-1 hover:bg-[var(--hover)]">
                      <span className={cn('h-2 w-2 shrink-0 rounded-full', p.label === 1 ? 'bg-[var(--success)]' : 'bg-[var(--danger)]')} />
                      <span className="flex-1 font-mono text-[11px] text-[var(--text-muted)]">{Math.round(p.x)}, {Math.round(p.y)}</span>
                      <span className="text-[10px] text-[var(--text-muted)]">{p.label === 1 ? '+' : '−'}</span>
                      <button type="button" onClick={() => dispatch(removePoint(i))} className="rounded p-0.5 text-[var(--text-muted)] hover:text-[var(--danger)]">
                        <IconX size={11} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>)}

          {/* ── Text mode controls ── */}
          {segMode === 'text' && (<>
            <section className="flex flex-col gap-1.5">
              <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Text Prompt</span>
              <textarea
                rows={3}
                value={textPrompt}
                onChange={(e) => dispatch(setTextPrompt(e.target.value))}
                placeholder="e.g. crack . peeling . mold"
                className="w-full resize-none rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[12px] text-[var(--text)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
              />
              <p className="text-[11px] text-[var(--text-muted)]">Separate multiple terms with <code className="font-mono"> . </code></p>
            </section>

            <section className="flex flex-col gap-1.5">
              <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">GroundingDINO Checkpoint</span>
              <div className="flex gap-1.5">
                <div
                  className="flex min-w-0 flex-1 items-center truncate rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[12px] text-[var(--text-muted)]"
                  title={gdinoCheckpoint || ''}
                >
                  <span className="truncate">{gdinoCheckpoint ? gdinoCheckpoint.split('/').pop() : 'Auto-detect'}</span>
                </div>
                <IconButton label="Browse checkpoint" onClick={browseGdino}>
                  <IconFolderOpen size={15} />
                </IconButton>
              </div>
              <p className="text-[11px] text-[var(--text-muted)]">Leave empty to use repo default.</p>
            </section>

            <section className="flex flex-col gap-1.5">
              <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Thresholds</span>
              <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-2 gap-y-1.5">
                <span className="text-[11px] text-[var(--text-muted)]">Box</span>
                <input
                  type="range"
                  min={0.05}
                  max={0.6}
                  step={0.01}
                  value={boxThreshold}
                  onChange={(e) => dispatch(setBoxThreshold(parseFloat(e.target.value)))}
                  className="accent-[var(--primary)]"
                />
                <span className="font-mono text-[11px] tabular-nums text-[var(--text)]">{boxThreshold.toFixed(2)}</span>

                <span className="text-[11px] text-[var(--text-muted)]">Text</span>
                <input
                  type="range"
                  min={0.05}
                  max={0.6}
                  step={0.01}
                  value={textThreshold}
                  onChange={(e) => dispatch(setTextThreshold(parseFloat(e.target.value)))}
                  className="accent-[var(--primary)]"
                />
                <span className="font-mono text-[11px] tabular-nums text-[var(--text)]">{textThreshold.toFixed(2)}</span>
              </div>
              <p className="text-[11px] text-[var(--text-muted)]">Lower = more (potentially noisy) detections.</p>
            </section>
          </>)}

          {/* Device */}
          <section className="flex flex-col gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">Device</span>
            <select
              value={device}
              onChange={(e) => dispatch(setDevice(e.target.value))}
              className="rounded-[5px] border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[12px] text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
            >
              {DEVICES.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </section>

          {/* Run button */}
          <Button
            variant="primary"
            onClick={() => dispatch(runSegmentation())}
            disabled={!canRun}
            className="flex items-center justify-center gap-2"
          >
            {running ? (
              <>
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[var(--primary)] border-t-transparent" />
                Running…
              </>
            ) : (
              <>
                <IconPlayerPlay size={14} />
                Run Segmentation
              </>
            )}
          </Button>

          {!canRun && !running && (
            <p className="text-[11px] text-[var(--text-muted)]">
              {!imagePath ? 'Select an image first.'
                : !samCheckpoint ? 'Select a SAM checkpoint.'
                : segMode === 'text' ? 'Enter a text prompt.'
                : 'Place at least one point.'}
            </p>
          )}

          {/* Error */}
          {status === 'error' && error && (
            <div className="rounded-[5px] border border-[var(--danger)] bg-[var(--danger-bg)] p-2.5 text-[11px] text-[var(--danger)]">
              {error.split('\n')[0]}
            </div>
          )}

          {/* Result info */}
          {status === 'done' && result && (
            <section className="flex flex-col gap-2 rounded-[5px] border border-[var(--border-muted)] bg-[var(--surface-2)] p-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-[var(--text)]">Result</span>
                <button
                  type="button"
                  onClick={() => dispatch(toggleOverlay())}
                  className="flex items-center gap-1.5 text-[11px] text-[var(--primary)] hover:underline"
                >
                  {showOverlay
                    ? <><IconToggleRight size={14} /> Overlay</>
                    : <><IconToggleLeft size={14} /> Original</>
                  }
                </button>
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                {result.score != null && (<>
                  <span className="text-[11px] text-[var(--text-muted)]">Score</span>
                  <span className="text-[11px] text-[var(--text)]">{result.score.toFixed(3)}</span>
                </>)}
                {result.detections != null && (<>
                  <span className="text-[11px] text-[var(--text-muted)]">Detections</span>
                  <span className="text-[11px] text-[var(--text)]">{result.detections}</span>
                </>)}
                {result.maskArea != null && (<>
                  <span className="text-[11px] text-[var(--text-muted)]">Mask area</span>
                  <span className="text-[11px] text-[var(--text)]">{result.maskArea?.toLocaleString()} px</span>
                </>)}
                <span className="text-[11px] text-[var(--text-muted)]">Model</span>
                <span className="text-[11px] text-[var(--text)]">{result.modelType}</span>
                <span className="text-[11px] text-[var(--text-muted)]">Device</span>
                <span className="text-[11px] text-[var(--text)]">{result.device}</span>
              </div>
            </section>
          )}
        </aside>

        {/* ── Right: canvas area ── */}
        <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[var(--surface-2)]">
          {!imagePath ? (
            <div className="flex flex-col items-center gap-3 text-center text-[var(--text-muted)]">
              <div className="flex h-14 w-14 items-center justify-center rounded-[8px] border border-[var(--border)] bg-[var(--surface)]">
                <IconFolderOpen size={28} strokeWidth={1.2} />
              </div>
              <p className="text-[13px]">Select an image to start</p>
              <Button variant="secondary" onClick={browseImage}>Browse Image</Button>
            </div>
          ) : (
            <PointCanvas
              imageSrc={canvasSrc}
              points={canvasPoints}
              mode={pointMode}
              onPointAdded={segMode === 'point' ? handlePointAdded : undefined}
            />
          )}

          {/* Mode badge */}
          {imagePath && (
            <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5 rounded-[4px] border border-[var(--primary)] bg-[var(--bg)] px-2 py-1 text-[11px] font-medium text-[var(--primary)]">
              {segMode === 'point'
                ? <><IconPointer size={12} /> Point</>
                : <><IconTextCaption size={12} /> Text</>
              }
            </div>
          )}

          {/* Running overlay */}
          {running && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg)] bg-opacity-60">
              <div className="flex flex-col items-center gap-3">
                <span className="h-7 w-7 animate-spin rounded-full border-2 border-[var(--primary)] border-t-transparent" />
                <span className="text-[12px] text-[var(--text-muted)]">
                  {segMode === 'text' ? 'Running GDino + SAM…' : 'Running SAM…'}
                </span>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
