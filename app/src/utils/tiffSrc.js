// TIFF (.tif/.tiff) support helpers.
// Browsers can't render TIFF natively, so we decode to a PNG data URL
// using the `utif` library. Non-TIFF sources pass through unchanged.

import { useEffect, useState } from 'react';
import UTIF from 'utif';

const tifCache = new Map();   // src -> dataUrl

export function isTifSrc(src) {
  if (!src || typeof src !== 'string') return false;
  if (src.startsWith('data:')) return false;
  const path = src.toLowerCase().split(/[?#]/)[0];
  return path.endsWith('.tif') || path.endsWith('.tiff');
}

export async function tifToDataUrl(src) {
  if (tifCache.has(src)) return tifCache.get(src);

  const resp = await fetch(src);
  if (!resp.ok) throw new Error(`Failed to fetch TIFF: ${resp.status}`);
  const buf = await resp.arrayBuffer();

  const ifds = UTIF.decode(buf);
  if (!ifds.length) throw new Error('No TIFF pages found');
  const ifd = ifds[0];
  UTIF.decodeImage(buf, ifd);
  const rgba = UTIF.toRGBA8(ifd);

  const canvas = document.createElement('canvas');
  canvas.width = ifd.width;
  canvas.height = ifd.height;
  const ctx = canvas.getContext('2d');
  const clamped = new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.byteLength);
  const data = new ImageData(clamped, ifd.width, ifd.height);
  ctx.putImageData(data, 0, 0);
  const dataUrl = canvas.toDataURL('image/png');

  tifCache.set(src, dataUrl);
  return dataUrl;
}

export async function resolveImageSrc(src) {
  if (!src) return src;
  if (isTifSrc(src)) return await tifToDataUrl(src);
  return src;
}

// Async helper: returns an HTMLImageElement once loaded (handles TIF + PNG/JPEG)
export async function loadImageAsync(src) {
  const resolved = await resolveImageSrc(src);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load: ${resolved}`));
    img.src = resolved;
  });
}

// React hook: pass a raw src (file:// or data:), get back a displayable URL
// (decoded data URL for TIFF, original src otherwise). Returns null while loading.
export function useResolvedImageSrc(src) {
  const initial = !src || isTifSrc(src) ? null : src;
  const [resolved, setResolved] = useState(initial);

  useEffect(() => {
    if (!src) { setResolved(null); return; }
    if (!isTifSrc(src)) { setResolved(src); return; }

    let cancelled = false;
    setResolved(null);
    tifToDataUrl(src)
      .then((s) => { if (!cancelled) setResolved(s); })
      .catch(() => {
        if (!cancelled) setResolved(null);
      });
    return () => { cancelled = true; };
  }, [src]);

  return resolved;
}
