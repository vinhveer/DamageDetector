import { useEffect, useRef, useState } from 'react';
import { drawCropToCanvas, getCropDisplaySize, normalizeBox } from '../imageCrop.js';

export default function ResultImage({ row, imageSize }) {
  const cropCanvasRef = useRef(null);
  const [status, setStatus] = useState('loading');
  const [cropDisplaySize, setCropDisplaySize] = useState({ width: imageSize, height: imageSize });

  useEffect(() => {
    if (!row.image_uri) {
      setStatus('no-image');
      return undefined;
    }

    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      const cropCanvas = cropCanvasRef.current;
      const box = normalizeBox(row, image.naturalWidth, image.naturalHeight);
      if (!cropCanvas || !box) {
        setCropDisplaySize({ width: imageSize, height: imageSize });
        setStatus('no-box');
        return;
      }

      const nextCropSize = getCropDisplaySize(box, imageSize);
      drawCropToCanvas(cropCanvas, image, box, nextCropSize.width, nextCropSize.height);
      setCropDisplaySize(nextCropSize);
      setStatus('ready');
    };
    image.onerror = () => {
      if (!cancelled) setStatus('error');
    };
    setStatus('loading');
    setCropDisplaySize({ width: imageSize, height: imageSize });
    image.src = row.image_uri;

    return () => { cancelled = true; };
  }, [imageSize, row]);

  if (!row.image_uri) {
    return (
      <div className="flex items-center justify-center rounded-[4px] bg-[var(--surface-2)] text-[12px] text-[var(--text-muted)]" style={{ width: imageSize, height: imageSize }}>
        No image
      </div>
    );
  }

  return (
    <div className="relative mx-auto overflow-hidden rounded-[4px] bg-[var(--surface-2)]" style={{ width: cropDisplaySize.width, height: cropDisplaySize.height }}>
        {status !== 'ready' && (
          <div className="absolute inset-0 flex items-center justify-center text-[12px] text-[var(--text-muted)]">
            {status === 'no-box' ? 'No box' : status === 'error' ? 'Image error' : 'Loading'}
          </div>
        )}
      <canvas ref={cropCanvasRef} className="absolute inset-0" aria-label={`crop ${row.image_rel_path || String(row.result_id)}`} />
    </div>
  );
}
