self.onmessage = async ({ data }) => {
  const { id, uri, bbox, size } = data || {};
  try {
    const response = await fetch(uri);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    const full = await createImageBitmap(blob);
    const [x1Raw, y1Raw, x2Raw, y2Raw] = bbox;
    const sourceWidth = full.width;
    const sourceHeight = full.height;
    const x1 = Math.max(0, Math.min(sourceWidth - 1, Number(x1Raw) || 0));
    const y1 = Math.max(0, Math.min(sourceHeight - 1, Number(y1Raw) || 0));
    const x2 = Math.max(x1 + 1, Math.min(sourceWidth, Number(x2Raw) || sourceWidth));
    const y2 = Math.max(y1 + 1, Math.min(sourceHeight, Number(y2Raw) || sourceHeight));
    const cropW = Math.max(1, x2 - x1);
    const cropH = Math.max(1, y2 - y1);
    const scale = size / Math.max(cropW, cropH);
    const drawW = cropW * scale;
    const drawH = cropH * scale;
    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(full, x1, y1, cropW, cropH, (size - drawW) / 2, (size - drawH) / 2, drawW, drawH);
    full.close?.();
    const bitmap = canvas.transferToImageBitmap();
    self.postMessage({ id, bitmap }, [bitmap]);
  } catch (error) {
    self.postMessage({ id, error: String(error?.message || error) });
  }
};
