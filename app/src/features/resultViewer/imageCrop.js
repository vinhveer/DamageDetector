export const normalizeBox = (row, naturalWidth, naturalHeight) => {
  let x1 = Number(row.x1);
  let y1 = Number(row.y1);
  let x2 = Number(row.x2);
  let y2 = Number(row.y2);

  if (![x1, y1, x2, y2, naturalWidth, naturalHeight].every(Number.isFinite) || naturalWidth <= 0 || naturalHeight <= 0) {
    return null;
  }

  if (x2 <= 1 && y2 <= 1) {
    x1 *= naturalWidth;
    x2 *= naturalWidth;
    y1 *= naturalHeight;
    y2 *= naturalHeight;
  }

  if (x2 <= x1 || y2 <= y1) return null;

  const x = Math.max(0, x1);
  const y = Math.max(0, y1);
  const right = Math.min(naturalWidth, x2);
  const bottom = Math.min(naturalHeight, y2);
  const width = right - x;
  const height = bottom - y;

  if (width <= 0 || height <= 0) return null;
  return { x, y, width, height };
};

export const drawCropToCanvas = (canvas, image, box, targetWidth, targetHeight) => {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(targetWidth));
  const height = Math.max(1, Math.round(targetHeight));
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const context = canvas.getContext('2d');
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(image, box.x, box.y, box.width, box.height, 0, 0, width, height);
};

export const getCropDisplaySize = (box, maxSize) => {
  const aspectRatio = box.width / box.height;
  return aspectRatio >= 1
    ? { width: maxSize, height: Math.max(1, Math.round(maxSize / aspectRatio)) }
    : { width: Math.max(1, Math.round(maxSize * aspectRatio)), height: maxSize };
};

export const createCropDataUrl = (row, maxSize = 1800) => new Promise((resolve) => {
  if (!row?.image_uri) {
    resolve('');
    return;
  }

  const image = new Image();
  image.onload = () => {
    const box = normalizeBox(row, image.naturalWidth, image.naturalHeight);
    if (!box) {
      resolve('');
      return;
    }
    const { width, height } = getCropDisplaySize(box, Math.min(maxSize, Math.max(box.width, box.height)));
    try {
      const canvas = document.createElement('canvas');
      drawCropToCanvas(canvas, image, box, width, height);
      resolve(canvas.toDataURL('image/png'));
    } catch {
      resolve('');
    }
  };
  image.onerror = () => resolve('');
  image.decoding = 'async';
  image.src = row.image_uri;
});

export const createCropObjectUrl = (row, maxSize = 1800, mimeType = 'image/png', quality) => new Promise((resolve) => {
  if (!row?.image_uri) {
    resolve('');
    return;
  }

  const image = new Image();
  image.onload = () => {
    const box = normalizeBox(row, image.naturalWidth, image.naturalHeight);
    if (!box) {
      resolve('');
      return;
    }
    const { width, height } = getCropDisplaySize(box, Math.min(maxSize, Math.max(box.width, box.height)));
    try {
      const canvas = document.createElement('canvas');
      drawCropToCanvas(canvas, image, box, width, height);
      canvas.toBlob((blob) => {
        resolve(blob ? URL.createObjectURL(blob) : canvas.toDataURL(mimeType, quality));
      }, mimeType, quality);
    } catch {
      resolve('');
    }
  };
  image.onerror = () => resolve('');
  image.decoding = 'async';
  image.src = row.image_uri;
});
