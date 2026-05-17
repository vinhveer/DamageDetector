import { useCallback, useEffect, useRef, useState } from 'react';
import { formatFloat } from '../utils.js';
import { createCropObjectUrl } from '../imageCrop.js';

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 8;
const ZOOM_STEP = 0.25;
const LIGHTBOX_CROP_MAX_SIZE = 1800;
const LIGHTBOX_CROP_MIME_TYPE = 'image/jpeg';
const LIGHTBOX_CROP_QUALITY = 0.92;
const LIGHTBOX_CROP_EXTENSION = 'jpg';
const PREFETCH_RADIUS = 8;
const PREFETCH_BATCH_SIZE = 4;
const PREFETCH_STAGGER_MS = 40;
const NEIGHBOR_PREFETCH_OFFSETS = Array.from({ length: PREFETCH_RADIUS }, (_, index) => [index + 1, -(index + 1)]).flat();

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function ToolButton({ children, disabled = false, onClick, title }) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className="inline-flex h-8 items-center justify-center rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-3 text-[13px] font-medium text-[var(--text)] hover:bg-[var(--hover)] disabled:cursor-not-allowed disabled:opacity-45"
    >
      {children}
    </button>
  );
}

async function createSlide(group, groupIndex) {
  const row = group?.[0];
  if (!row) return null;

  const src = await createCropObjectUrl(row, LIGHTBOX_CROP_MAX_SIZE, LIGHTBOX_CROP_MIME_TYPE, LIGHTBOX_CROP_QUALITY);
  return {
    src,
    download: src,
    extension: LIGHTBOX_CROP_EXTENSION,
    title: `#${groupIndex + 1} · id ${row.result_id}`,
    description: `${row.predicted_label || '-'} · conf=${formatFloat(row.predicted_probability_pct, 1)}% · dist=${formatFloat(row.distance_to_center)}`,
    groupIndex
  };
}

function revokeSlides(slidesByIndex) {
  Object.values(slidesByIndex).forEach((slide) => {
    if (slide?.src?.startsWith('blob:')) URL.revokeObjectURL(slide.src);
  });
}

export default function ImageLightbox({ groups, index, onClose }) {
  const [slidesByIndex, setSlidesByIndex] = useState({});
  const [activeIndex, setActiveIndex] = useState(index);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const slidesRef = useRef({});
  const loadingRef = useRef(new Set());
  const generationRef = useRef(0);
  const dragRef = useRef(null);

  const open = index >= 0;

  useEffect(() => {
    generationRef.current += 1;
    loadingRef.current.clear();
    slidesRef.current = {};
    setSlidesByIndex((current) => {
      revokeSlides(current);
      return {};
    });
  }, [groups]);

  useEffect(() => {
    if (open) return undefined;

    generationRef.current += 1;
    loadingRef.current.clear();
    slidesRef.current = {};
    setSlidesByIndex((current) => {
      revokeSlides(current);
      return {};
    });
    return undefined;
  }, [open]);

  useEffect(() => () => revokeSlides(slidesRef.current), []);

  useEffect(() => {
    if (index >= 0) setActiveIndex(index);
  }, [index]);

  const loadSlide = useCallback((groupIndex) => {
    if (groupIndex < 0 || groupIndex >= groups.length) return undefined;
    if (slidesRef.current[groupIndex] || loadingRef.current.has(groupIndex)) return undefined;

    const group = groups[groupIndex];
    if (!group?.[0]) return undefined;

    const generation = generationRef.current;
    loadingRef.current.add(groupIndex);

    return createSlide(group, groupIndex)
      .then((nextSlide) => {
        if (generation !== generationRef.current || !nextSlide) {
          revokeSlides(nextSlide ? { [groupIndex]: nextSlide } : {});
          return;
        }
        setSlidesByIndex((current) => {
          if (current[groupIndex]) {
            revokeSlides({ [groupIndex]: nextSlide });
            return current;
          }
          const next = { ...current, [groupIndex]: nextSlide };
          slidesRef.current = next;
          return next;
        });
      })
      .finally(() => {
        if (generation !== generationRef.current) return;
        loadingRef.current.delete(groupIndex);
      });
  }, [groups]);

  useEffect(() => {
    if (!open) return undefined;

    loadSlide(activeIndex);
    const timers = [];
    for (let indexOffset = 0; indexOffset < NEIGHBOR_PREFETCH_OFFSETS.length; indexOffset += PREFETCH_BATCH_SIZE) {
      const batch = NEIGHBOR_PREFETCH_OFFSETS.slice(indexOffset, indexOffset + PREFETCH_BATCH_SIZE);
      const timer = window.setTimeout(() => {
        batch.forEach((prefetchOffset) => loadSlide(activeIndex + prefetchOffset));
      }, PREFETCH_STAGGER_MS * (indexOffset / PREFETCH_BATCH_SIZE + 1));
      timers.push(timer);
    }

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [activeIndex, loadSlide, open]);

  useEffect(() => {
    setZoom(1);
    setRotation(0);
    setOffset({ x: 0, y: 0 });
  }, [activeIndex]);

  const activeSlide = slidesByIndex[activeIndex];
  const activeRow = groups[activeIndex]?.[0];
  const activeReady = Boolean(activeSlide?.src);
  const canGoPrevious = activeIndex > 0;
  const canGoNext = activeIndex >= 0 && activeIndex < groups.length - 1;
  const meta = activeSlide || (activeRow ? {
    title: `#${activeIndex + 1} · id ${activeRow.result_id}`,
    description: `${activeRow.predicted_label || '-'} · conf=${formatFloat(activeRow.predicted_probability_pct, 1)}% · dist=${formatFloat(activeRow.distance_to_center)}`
  } : { title: '', description: '' });

  useEffect(() => {
    if (!open) return undefined;

    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
      if (event.key === 'ArrowLeft' && canGoPrevious) setActiveIndex((value) => value - 1);
      if (event.key === 'ArrowRight' && canGoNext) setActiveIndex((value) => value + 1);
      if ((event.key === '+' || event.key === '=') && activeReady) setZoom((value) => clamp(value + ZOOM_STEP, MIN_ZOOM, MAX_ZOOM));
      if (event.key === '-' && activeReady) setZoom((value) => clamp(value - ZOOM_STEP, MIN_ZOOM, MAX_ZOOM));
      if (event.key === '0' && activeReady) {
        setZoom(1);
        setRotation(0);
        setOffset({ x: 0, y: 0 });
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeReady, canGoNext, canGoPrevious, onClose, open]);

  if (!open) return null;

  const goPrevious = () => {
    if (canGoPrevious) setActiveIndex((value) => value - 1);
  };

  const goNext = () => {
    if (canGoNext) setActiveIndex((value) => value + 1);
  };

  const resetView = () => {
    setZoom(1);
    setRotation(0);
    setOffset({ x: 0, y: 0 });
  };

  const toggleFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
      return;
    }
    document.documentElement.requestFullscreen?.();
  };

  const onPointerDown = (event) => {
    if (!activeReady) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      offset
    };
  };

  const onPointerMove = (event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    setOffset({
      x: drag.offset.x + event.clientX - drag.startX,
      y: drag.offset.y + event.clientY - drag.startY
    });
  };

  const onPointerUp = (event) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  const onWheel = (event) => {
    if (!activeReady) return;
    event.preventDefault();
    setZoom((value) => clamp(value + (event.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP), MIN_ZOOM, MAX_ZOOM));
  };

  return (
    <div className="fixed inset-0 z-[9999] flex bg-[var(--bg)] text-[var(--text)] rv-font">
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-4 border-b border-[var(--border-muted)] bg-[var(--surface)] px-4">
          <div className="min-w-0 flex-1">
            <div className="truncate text-[13px] font-semibold text-[var(--text)]">{meta.title || 'Image crop'}</div>
            <div className="truncate text-[12px] text-[var(--text-muted)]">{meta.description || 'Preparing crop'}</div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <ToolButton onClick={goPrevious} disabled={!canGoPrevious} title="Previous">Prev</ToolButton>
            <ToolButton onClick={goNext} disabled={!canGoNext} title="Next">Next</ToolButton>
            <ToolButton onClick={onClose} title="Close">Close</ToolButton>
          </div>
        </header>

        <div className="flex h-11 shrink-0 items-center justify-between gap-3 overflow-x-auto border-b border-[var(--border-muted)] bg-[var(--bg)] px-4">
          <div className="flex shrink-0 items-center gap-2">
            <ToolButton onClick={() => setZoom((value) => clamp(value - ZOOM_STEP, MIN_ZOOM, MAX_ZOOM))} disabled={!activeReady} title="Zoom out">-</ToolButton>
            <div className="w-16 text-center text-[13px] text-[var(--text-muted)]">{Math.round(zoom * 100)}%</div>
            <ToolButton onClick={() => setZoom((value) => clamp(value + ZOOM_STEP, MIN_ZOOM, MAX_ZOOM))} disabled={!activeReady} title="Zoom in">+</ToolButton>
            <ToolButton onClick={() => setRotation((value) => value - 90)} disabled={!activeReady} title="Rotate left">Rotate L</ToolButton>
            <ToolButton onClick={() => setRotation((value) => value + 90)} disabled={!activeReady} title="Rotate right">Rotate R</ToolButton>
            <ToolButton onClick={resetView} disabled={!activeReady} title="Reset view">Reset</ToolButton>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {activeReady && (
              <a
                href={activeSlide.src}
                download={`crop-${activeSlide.groupIndex + 1}.${activeSlide.extension || 'png'}`}
                className="inline-flex h-8 items-center justify-center rounded-[6px] border border-[var(--border)] bg-[var(--surface-2)] px-3 text-[13px] font-medium text-[var(--text)] hover:bg-[var(--hover)]"
              >
                Download
              </a>
            )}
            <ToolButton onClick={toggleFullscreen} title="Browser fullscreen">Fullscreen</ToolButton>
          </div>
        </div>

        <main
          className="relative min-h-0 flex-1 touch-none overflow-hidden bg-[var(--surface-2)]"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
        >
          {!activeSlide && (
            <div className="absolute inset-0 flex items-center justify-center text-[13px] text-[var(--text-muted)]">
              Preparing cropped image
            </div>
          )}
          {activeSlide && !activeReady && (
            <div className="absolute inset-0 flex items-center justify-center text-[13px] text-[var(--text-muted)]">
              Crop unavailable
            </div>
          )}
          {activeReady && (
            <div className="absolute inset-0 flex items-center justify-center cursor-grab active:cursor-grabbing">
              <img
                src={activeSlide.src}
                alt={meta.title}
                draggable="false"
                className="max-h-[82vh] max-w-[88vw] select-none object-contain"
                style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom}) rotate(${rotation}deg)` }}
              />
            </div>
          )}
        </main>

        <footer className="flex h-24 shrink-0 items-center gap-2 overflow-x-auto border-t border-[var(--border-muted)] bg-[var(--surface)] px-4">
          {groups.length === 0 && <div className="text-[13px] text-[var(--text-muted)]">No crops</div>}
          {groups.map((group, groupIndex) => {
            const slide = slidesByIndex[groupIndex];
            const selected = groupIndex === activeIndex;
            const title = slide?.title || `#${groupIndex + 1} · id ${group?.[0]?.result_id || '-'}`;
            return (
              <button
                type="button"
                key={`${group?.[0]?.result_id || 'crop'}-${groupIndex}`}
                onClick={() => setActiveIndex(groupIndex)}
                className={`h-16 w-20 shrink-0 overflow-hidden rounded-[6px] border bg-[var(--surface-2)] ${selected ? 'border-[var(--primary)]' : 'border-[var(--border-muted)] hover:border-[var(--border)]'}`}
                title={title}
              >
                {slide?.src ? (
                  <img src={slide.src} alt={title} className="h-full w-full object-contain" draggable="false" />
                ) : (
                  <span className="flex h-full w-full items-center justify-center text-[12px] text-[var(--text-muted)]">#{groupIndex + 1}</span>
                )}
              </button>
            );
          })}
        </footer>
      </div>
    </div>
  );
}
