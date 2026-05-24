import { useEffect, useRef, useState } from 'react';
import { getBitmap } from '../../../utils/imageCache.js';

export default function BoxThumbnail({ box, size = 200, showLabel = true }) {
  const canvasRef = useRef(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setError(false);
    if (!box?.image_uri) {
      setError(true);
      return;
    }
    getBitmap(box.image_uri)
      .then((bitmap) => {
        if (cancelled) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const dpr = window.devicePixelRatio || 1;
        const pixel = Math.max(64, size * dpr);
        canvas.width = pixel;
        canvas.height = pixel;
        canvas.style.width = `${size}px`;
        canvas.style.height = `${size}px`;
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, pixel, pixel);

        const margin = Math.max(6, pixel * 0.04);
        const bx1 = Math.max(0, Math.min(box.image_width, box.x1));
        const by1 = Math.max(0, Math.min(box.image_height, box.y1));
        const bx2 = Math.max(0, Math.min(box.image_width, box.x2));
        const by2 = Math.max(0, Math.min(box.image_height, box.y2));
        const bw = Math.max(1, bx2 - bx1);
        const bh = Math.max(1, by2 - by1);
        const cx1 = Math.max(0, bx1 - bw * 0.3);
        const cy1 = Math.max(0, by1 - bh * 0.3);
        const cx2 = Math.min(box.image_width, bx2 + bw * 0.3);
        const cy2 = Math.min(box.image_height, by2 + bh * 0.3);
        const cw = Math.max(1, cx2 - cx1);
        const ch = Math.max(1, cy2 - cy1);
        const scale = Math.min((pixel - margin * 2) / cw, (pixel - margin * 2) / ch);
        const dw = cw * scale;
        const dh = ch * scale;
        const dx = (pixel - dw) / 2;
        const dy = (pixel - dh) / 2;
        ctx.drawImage(bitmap, cx1, cy1, cw, ch, dx, dy, dw, dh);

        const ox = dx + (bx1 - cx1) * scale;
        const oy = dy + (by1 - cy1) * scale;
        const ow = bw * scale;
        const oh = bh * scale;
        ctx.lineWidth = Math.max(2, pixel / 100);
        ctx.strokeStyle = box.predicted_label === 'crack' ? '#fbbf24'
          : box.predicted_label === 'mold' ? '#10b981'
          : box.predicted_label === 'spall' ? '#60a5fa'
          : '#f472b6';
        ctx.strokeRect(ox, oy, ow, oh);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => { cancelled = true; };
  }, [box, size]);

  return (
    <div className="group relative overflow-hidden rounded-[6px] bg-black ring-1 ring-[var(--border)]">
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size }}
        className="block"
      />
      {showLabel && (
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between bg-gradient-to-t from-black/85 via-black/55 to-transparent px-2 py-1.5 text-[10px] font-mono text-white">
          <span>#{box.result_id}</span>
          <span className="uppercase tracking-wide">{box.predicted_label}</span>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center text-[11px] text-[var(--danger)]">
          no image
        </div>
      )}
    </div>
  );
}
