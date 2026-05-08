import { useEffect, useMemo, useRef, useState } from 'react';
import { formatFloat } from '../utils.js';
import { createCropDataUrl } from '../imageCrop.js';

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 8;
const ZOOM_STEP = 0.25;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function ToolButton({ children, disabled = false, onClick, title }) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className="inline-flex h-8 items-center justify-center rounded-md border border-[var(--docker-border)] bg-white px-3 text-[13px] font-medium text-[var(--docker-text)] hover:bg-[var(--docker-hover)] disabled:cursor-not-allowed disabled:opacity-45"
    >
      {children}
    </button>
  );
}

export default function ImageLightbox({ groups, index, onClose }) {
  const [slides, setSlides] = useState([]);
  const [activeIndex, setActiveIndex] = useState(index);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragRef = useRef(null);

  const open = index >= 0;

  useEffect(() => {
    if (index < 0) {
      setSlides([]);
      return undefined;
    }

    setActiveIndex(index);
    let cancelled = false;
    Promise.all(groups.map(async (group, groupIndex) => {
      const row = group[0];
      const src = await createCropDataUrl(row);
      if (!src) return null;
      return {
        src,
        download: src,
        title: `#${groupIndex + 1} · id ${row.result_id}`,
        description: `${row.predicted_label || '-'} · conf=${formatFloat(row.predicted_probability_pct, 1)}% · dist=${formatFloat(row.distance_to_center)}`,
        groupIndex
      };
    })).then((nextSlides) => {
      if (!cancelled) setSlides(nextSlides.filter(Boolean));
    });
    return () => { cancelled = true; };
  }, [groups, index]);

  useEffect(() => {
    setZoom(1);
    setRotation(0);
    setOffset({ x: 0, y: 0 });
  }, [activeIndex]);

  const slideIndex = slides.findIndex((slide) => slide.groupIndex === activeIndex);
  const activeSlide = slides[slideIndex];
  const canGoPrevious = slideIndex > 0;
  const canGoNext = slideIndex >= 0 && slideIndex < slides.length - 1;

  const meta = useMemo(() => {
    if (!activeSlide) return { title: '', description: '' };
    return {
      title: activeSlide.title,
      description: activeSlide.description
    };
  }, [activeSlide]);

  useEffect(() => {
    if (open && slides.length > 0 && slideIndex < 0) setActiveIndex(slides[0].groupIndex);
  }, [open, slideIndex, slides]);

  useEffect(() => {
    if (!open) return undefined;

    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
      if (event.key === 'ArrowLeft' && canGoPrevious) setActiveIndex(slides[slideIndex - 1].groupIndex);
      if (event.key === 'ArrowRight' && canGoNext) setActiveIndex(slides[slideIndex + 1].groupIndex);
      if ((event.key === '+' || event.key === '=') && activeSlide) setZoom((value) => clamp(value + ZOOM_STEP, MIN_ZOOM, MAX_ZOOM));
      if (event.key === '-' && activeSlide) setZoom((value) => clamp(value - ZOOM_STEP, MIN_ZOOM, MAX_ZOOM));
      if (event.key === '0' && activeSlide) {
        setZoom(1);
        setRotation(0);
        setOffset({ x: 0, y: 0 });
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeSlide, canGoNext, canGoPrevious, onClose, open, slideIndex, slides]);

  if (!open) return null;

  const goPrevious = () => {
    if (canGoPrevious) setActiveIndex(slides[slideIndex - 1].groupIndex);
  };

  const goNext = () => {
    if (canGoNext) setActiveIndex(slides[slideIndex + 1].groupIndex);
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
    if (!activeSlide) return;
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
    if (!activeSlide) return;
    event.preventDefault();
    setZoom((value) => clamp(value + (event.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP), MIN_ZOOM, MAX_ZOOM));
  };

  return (
    <div className="fixed inset-0 z-[9999] flex bg-white text-[var(--docker-text)] rv-font">
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center justify-between border-b border-[var(--docker-border-soft)] bg-white px-4">
          <div className="min-w-0">
            <div className="truncate text-[13px] font-semibold">{meta.title || 'Image crop'}</div>
            <div className="truncate text-[12px] text-[var(--docker-muted)]">{meta.description || 'Preparing crop'}</div>
          </div>
          <div className="flex items-center gap-2">
            <ToolButton onClick={goPrevious} disabled={!canGoPrevious} title="Previous">Prev</ToolButton>
            <ToolButton onClick={goNext} disabled={!canGoNext} title="Next">Next</ToolButton>
            <ToolButton onClick={onClose} title="Close">Close</ToolButton>
          </div>
        </header>

        <div className="flex h-11 shrink-0 items-center justify-between gap-3 overflow-x-auto border-b border-[var(--docker-border-soft)] bg-[var(--docker-bg)] px-4">
          <div className="flex shrink-0 items-center gap-2">
            <ToolButton onClick={() => setZoom((value) => clamp(value - ZOOM_STEP, MIN_ZOOM, MAX_ZOOM))} disabled={!activeSlide} title="Zoom out">-</ToolButton>
            <div className="w-16 text-center text-[13px] text-[var(--docker-muted)]">{Math.round(zoom * 100)}%</div>
            <ToolButton onClick={() => setZoom((value) => clamp(value + ZOOM_STEP, MIN_ZOOM, MAX_ZOOM))} disabled={!activeSlide} title="Zoom in">+</ToolButton>
            <ToolButton onClick={() => setRotation((value) => value - 90)} disabled={!activeSlide} title="Rotate left">Rotate L</ToolButton>
            <ToolButton onClick={() => setRotation((value) => value + 90)} disabled={!activeSlide} title="Rotate right">Rotate R</ToolButton>
            <ToolButton onClick={resetView} disabled={!activeSlide} title="Reset view">Reset</ToolButton>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {activeSlide && (
              <a
                href={activeSlide.src}
                download={`crop-${activeSlide.groupIndex + 1}.png`}
                className="inline-flex h-8 items-center justify-center rounded-md border border-[var(--docker-border)] bg-white px-3 text-[13px] font-medium text-[var(--docker-text)] hover:bg-[var(--docker-hover)]"
              >
                Download
              </a>
            )}
            <ToolButton onClick={toggleFullscreen} title="Browser fullscreen">Fullscreen</ToolButton>
          </div>
        </div>

        <main
          className="relative min-h-0 flex-1 touch-none overflow-hidden bg-[#f4f5f7]"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
        >
          {!activeSlide && (
            <div className="absolute inset-0 flex items-center justify-center text-[13px] text-[var(--docker-muted)]">
              Preparing cropped image
            </div>
          )}
          {activeSlide && (
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

        <footer className="flex h-24 shrink-0 items-center gap-2 overflow-x-auto border-t border-[var(--docker-border-soft)] bg-white px-4">
          {slides.length === 0 && <div className="text-[13px] text-[var(--docker-muted)]">Loading crops</div>}
          {slides.map((slide) => {
            const selected = slide.groupIndex === activeIndex;
            return (
              <button
                type="button"
                key={slide.groupIndex}
                onClick={() => setActiveIndex(slide.groupIndex)}
                className={`h-16 w-20 shrink-0 overflow-hidden rounded-md border bg-white ${selected ? 'border-[var(--docker-blue)]' : 'border-[var(--docker-border-soft)] hover:border-[var(--docker-border)]'}`}
                title={slide.title}
              >
                <img src={slide.src} alt={slide.title} className="h-full w-full object-contain" draggable="false" />
              </button>
            );
          })}
        </footer>
      </div>
    </div>
  );
}
