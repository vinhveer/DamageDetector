import { useCallback, useEffect, useRef } from 'react';
import { resolveImageSrc } from '../../utils/tiffSrc.js';

const LABEL_COLORS = { crack: '#f85149', mold: '#3fb950', spall: '#d29922' };
const defaultColor = '#58a6ff';

function labelColor(label) { return LABEL_COLORS[label?.toLowerCase()] ?? defaultColor; }

export default function BoxOverlay({ imageSrc, boxes = [], suspectBoxes = [], showSuspect = false }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const img = imgRef.current;
    if (!img.naturalWidth) return;

    const dpr = window.devicePixelRatio || 1;
    const cw = container.clientWidth, ch = container.clientHeight;
    canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
    canvas.style.width = cw + 'px'; canvas.style.height = ch + 'px';

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight, 1);
    const iw = img.naturalWidth * scale, ih = img.naturalHeight * scale;
    const ox = Math.floor((cw - iw) / 2), oy = Math.floor((ch - ih) / 2);
    ctx.drawImage(img, ox, oy, iw, ih);

    const drawBoxes = (bxs, alpha) => {
      bxs.forEach(b => {
        const x = ox + b.x1 * scale, y = oy + b.y1 * scale;
        const w = (b.x2 - b.x1) * scale, h = (b.y2 - b.y1) * scale;
        const c = labelColor(b.label);
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = c; ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.globalAlpha = alpha * 0.15;
        ctx.fillStyle = c; ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;
        if (b.score != null) {
          ctx.fillStyle = c; ctx.font = '11px system-ui';
          ctx.fillText(`${b.label} ${(b.score * 100).toFixed(0)}%`, x + 2, y - 3);
        }
      });
    };

    drawBoxes(boxes, 1.0);
    if (showSuspect) drawBoxes(suspectBoxes, 0.55);
  }, [boxes, showSuspect, suspectBoxes]);

  useEffect(() => {
    if (!imageSrc) return;
    let cancelled = false;
    resolveImageSrc(imageSrc).then((resolved) => {
      if (cancelled) return;
      const img = imgRef.current;
      img.onload = draw;
      img.src = resolved;
    }).catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [draw, imageSrc]);

  useEffect(draw, [draw]);

  return (
    <div ref={containerRef} className="relative h-full w-full">
      <canvas ref={canvasRef} className="block h-full w-full" />
    </div>
  );
}
