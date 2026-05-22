import { useCallback, useEffect, useRef } from 'react';
import { resolveImageSrc } from '../../utils/tiffSrc.js';

export default function MaskOverlay({ imageSrc, masks = [], alpha = 0.55, highlightId = null }) {
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

    masks.forEach(m => {
      if (!m.mask_png_b64) return;
      const isHighlight = highlightId != null && m.detection_id === highlightId;
      const maskImg = new Image();
      maskImg.onload = () => {
        ctx.globalAlpha = isHighlight ? Math.min(alpha + 0.2, 1) : alpha;
        ctx.drawImage(maskImg, ox, oy, iw, ih);
        ctx.globalAlpha = 1;
      };
      maskImg.src = `data:image/png;base64,${m.mask_png_b64}`;
    });
  }, [alpha, highlightId, masks]);

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
