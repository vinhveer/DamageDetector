// Diverse subset selection for the review queue.
//
// Goal: instead of labeling every box, pick a small, representative sample
// (~10%) that is as DISTINCT as possible. We use the cached DINOv2 tight
// embeddings and run farthest-point sampling (FPS) per class so each class is
// covered and the picks spread across the embedding space (low redundancy).

// Decode a SQLite BLOB (Uint8Array/Buffer) into a Float32Array copy.
const decodeVec = (blob) => {
  if (!blob) return null;
  const u8 = blob instanceof Uint8Array ? blob : new Uint8Array(blob);
  // copy so we don't alias a pooled buffer
  const copy = u8.slice();
  return new Float32Array(copy.buffer, copy.byteOffset, Math.floor(copy.byteLength / 4));
};

// cosine distance for L2-normalized vectors: 1 - dot
const cosineDist = (a, b) => {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) dot += a[i] * b[i];
  return 1 - dot;
};

// Farthest-point sampling within one class group.
// items: [{ resultId, vec: Float32Array, reliability: number }]
// returns: array of selected resultIds
const fpsSelect = (items, k) => {
  const total = items.length;
  if (k >= total) return items.map((it) => it.resultId);
  if (k <= 0) return [];

  // seed with the most-uncertain item (lowest reliability), ties by result_id
  let seedIdx = 0;
  for (let i = 1; i < total; i += 1) {
    const a = items[i];
    const b = items[seedIdx];
    if (a.reliability < b.reliability || (a.reliability === b.reliability && a.resultId < b.resultId)) {
      seedIdx = i;
    }
  }

  const selected = [seedIdx];
  const minDist = new Float64Array(total);
  for (let i = 0; i < total; i += 1) {
    minDist[i] = items[i].vec ? cosineDist(items[i].vec, items[seedIdx].vec) : Infinity;
  }
  minDist[seedIdx] = -1; // mark selected

  while (selected.length < k) {
    // pick the point with the largest distance to the selected set
    let best = -1;
    let bestDist = -Infinity;
    for (let i = 0; i < total; i += 1) {
      if (minDist[i] > bestDist) {
        bestDist = minDist[i];
        best = i;
      }
    }
    if (best < 0) break;
    selected.push(best);
    const bv = items[best].vec;
    minDist[best] = -1;
    if (bv) {
      for (let i = 0; i < total; i += 1) {
        if (minDist[i] < 0) continue;
        const d = items[i].vec ? cosineDist(items[i].vec, bv) : Infinity;
        if (d < minDist[i]) minDist[i] = d;
      }
    }
  }

  return selected.map((i) => items[i].resultId);
};

// Stratified diverse sample across class groups.
// rows: [{ resultId, label, reliability, vec }]
// ratio: 0..1 fraction to keep (per class, rounded up, min 1 per non-empty class)
// returns: Set of selected resultIds
export const selectDiverseSample = (rows, ratio) => {
  const r = Number(ratio);
  if (!Number.isFinite(r) || r <= 0 || r >= 1) {
    return new Set(rows.map((row) => row.resultId));
  }
  const byLabel = new Map();
  for (const row of rows) {
    const key = row.label || 'unknown';
    if (!byLabel.has(key)) byLabel.set(key, []);
    byLabel.get(key).push(row);
  }
  const picked = new Set();
  for (const [, items] of byLabel) {
    const k = Math.max(1, Math.round(items.length * r));
    for (const id of fpsSelect(items, k)) picked.add(id);
  }
  return picked;
};

export { decodeVec };
