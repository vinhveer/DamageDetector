import { useEffect, useLayoutEffect, useRef, useState } from 'react';

// Renders a full image with the review box drawn on top. The box coords are in
// the image's native pixel space, so we scale them to the rendered size.
export default function BoxImage({ imageUri, cropUri, box }) {
  const wrapRef = useRef(null);
  const imgRef = useRef(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [render, setRender] = useState({ w: 0, h: 0, offX: 0, offY: 0 });

  // measure rendered image rect relative to its container (object-contain)
  const measure = () => {
    const wrap = wrapRef.current;
    const img = imgRef.current;
    if (!wrap || !img || !natural.w || !natural.h) return;
    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    const scale = Math.min(cw / natural.w, ch / natural.h);
    const w = natural.w * scale;
    const h = natural.h * scale;
    setRender({ w, h, offX: (cw - w) / 2, offY: (ch - h) / 2 });
  };

  useLayoutEffect(measure, [natural]);

  useEffect(() => {
    const onResize = () => measure();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [natural]); // eslint-disable-line react-hooks/exhaustive-deps

  const onLoad = (e) => {
    const t = e.currentTarget;
    setNatural({ w: t.naturalWidth, h: t.naturalHeight });
  };

  const scaleX = natural.w ? render.w / natural.w : 1;
  const scaleY = natural.h ? render.h / natural.h : 1;

  const overlay = box && natural.w
    ? {
        left: render.offX + box.x1 * scaleX,
        top: render.offY + box.y1 * scaleY,
        width: Math.max(1, (box.x2 - box.x1) * scaleX),
        height: Math.max(1, (box.y2 - box.y1) * scaleY),
      }
    : null;

  if (!imageUri) {
    return (
      <div className="flex h-full items-center justify-center rounded-[6px] border border-[var(--border-muted)] bg-[var(--surface-2)] text-[13px] text-[var(--text-muted)]">
        {cropUri ? (
          <img src={cropUri} alt="crop" className="max-h-full max-w-full object-contain" />
        ) : (
          'Không tìm thấy ảnh gốc (kiểm tra Image root).'
        )}
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 gap-3">
      <div
        ref={wrapRef}
        className="relative min-h-0 flex-1 overflow-hidden rounded-[6px] border border-[var(--border-muted)] bg-[var(--surface-2)]"
      >
        <img
          ref={imgRef}
          src={imageUri}
          alt="review"
          onLoad={onLoad}
          className="absolute inset-0 h-full w-full object-contain"
          draggable={false}
        />
        {overlay && (
          <div
            className="pointer-events-none absolute rounded-[2px]"
            style={{
              left: `${overlay.left}px`,
              top: `${overlay.top}px`,
              width: `${overlay.width}px`,
              height: `${overlay.height}px`,
              border: '2px solid var(--accent, #f43f5e)',
              boxShadow: '0 0 0 9999px rgba(0,0,0,0.45)',
            }}
          />
        )}
      </div>

      {cropUri && (
        <div className="flex w-[160px] shrink-0 flex-col gap-1">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">crop</div>
          <div className="flex flex-1 items-center justify-center overflow-hidden rounded-[6px] border border-[var(--border-muted)] bg-[var(--surface-2)] p-1">
            <img src={cropUri} alt="crop" className="max-h-full max-w-full object-contain" draggable={false} />
          </div>
        </div>
      )}
    </div>
  );
}
