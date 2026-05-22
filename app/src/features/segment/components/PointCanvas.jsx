import { useRef, useEffect, useCallback, useState } from 'react';
import { resolveImageSrc } from '../../../utils/tiffSrc.js';

const POINT_RADIUS = 6;

export default function PointCanvas({ imageSrc, points, mode, onPointAdded }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const transformRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
  const [imgReady, setImgReady] = useState(false);

  // Load image when src changes (TIFF is decoded to PNG data URL first)
  useEffect(() => {
    if (!imageSrc) return;
    setImgReady(false);
    let cancelled = false;
    resolveImageSrc(imageSrc).then((resolved) => {
      if (cancelled) return;
      const img = imgRef.current;
      img.onload = () => { if (!cancelled) setImgReady(true); };
      img.onerror = () => { if (!cancelled) setImgReady(false); };
      img.src = resolved;
    }).catch(() => { if (!cancelled) setImgReady(false); });
    return () => { cancelled = true; };
  }, [imageSrc]);

  // Redraw canvas whenever image or points change
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const cw = container.clientWidth;
    const ch = container.clientHeight;

    // Resize canvas buffer to container
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
      canvas.style.width = cw + 'px';
      canvas.style.height = ch + 'px';
    }

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    if (!imgReady) return;

    const img = imgRef.current;
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight, 1);
    const imgW = img.naturalWidth * scale;
    const imgH = img.naturalHeight * scale;
    const offsetX = Math.floor((cw - imgW) / 2);
    const offsetY = Math.floor((ch - imgH) / 2);
    transformRef.current = { scale, offsetX, offsetY };

    ctx.drawImage(img, offsetX, offsetY, imgW, imgH);

    // Draw points
    for (const p of points) {
      const cx = offsetX + p.x * scale;
      const cy = offsetY + p.y * scale;
      const isPos = p.label === 1;

      // Filled circle
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = isPos ? '#3fb950' : '#f85149';
      ctx.fill();

      // White border
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Cross-hair inside
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 3, cy);
      ctx.lineTo(cx + 3, cy);
      ctx.moveTo(cx, cy - 3);
      ctx.lineTo(cx, cy + 3);
      ctx.stroke();
    }
  }, [imgReady, imageSrc, points]);

  // Handle container resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => {
      // Trigger redraw by forcing a re-render — imgReady state doesn't change
      // but we can dispatch a custom event to force re-draw
      if (canvasRef.current) {
        canvasRef.current.dispatchEvent(new Event('resize-redraw'));
      }
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Listen for resize-redraw to re-trigger useEffect
  const [resizeTick, setResizeTick] = useState(0);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handler = () => setResizeTick((t) => t + 1);
    canvas.addEventListener('resize-redraw', handler);
    return () => canvas.removeEventListener('resize-redraw', handler);
  }, []);

  // Click: convert canvas coords → image coords → emit
  const handleClick = useCallback((e) => {
    if (!imgReady) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { scale, offsetX, offsetY } = transformRef.current;
    const imgX = (cx - offsetX) / scale;
    const imgY = (cy - offsetY) / scale;
    const img = imgRef.current;
    if (imgX < 0 || imgY < 0 || imgX > img.naturalWidth || imgY > img.naturalHeight) return;
    onPointAdded({ x: imgX, y: imgY, label: mode === 'positive' ? 1 : 0 });
  }, [imgReady, mode, onPointAdded]);

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full"
      style={{ cursor: imgReady ? 'crosshair' : 'default' }}
      onClick={handleClick}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      {/* invisible dep to trigger redraw on resize */}
      <span data-tick={resizeTick} className="hidden" />
    </div>
  );
}
