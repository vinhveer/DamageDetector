import { getBitmap } from './imageCache.js';

const SOURCE_CACHE_LIMIT = 200;
const THUMB_CACHE_LIMIT = 1500;
const WORKER_COUNT = Math.min(8, Math.max(2, navigator.hardwareConcurrency || 4));

const sourceCache = new Map();
const thumbCache = new Map();
const queued = [];
const workers = [];
const inflight = new Map();
let nextId = 1;
let activeFallback = 0;
let queueWarningTimer = null;
const counters = { cacheHit: 0, cacheMiss: 0, completed: 0, failed: 0 };

const canUseWorkers = typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined';

function keyFor({ uri, bbox, size }) {
  return `${uri}|${bbox.map((v) => Math.round(Number(v) || 0)).join(',')}|${size}`;
}

function touch(map, key, value) {
  map.delete(key);
  map.set(key, value);
}

function trim(map, limit) {
  while (map.size > limit) {
    const oldestKey = map.keys().next().value;
    const oldest = map.get(oldestKey);
    oldest?.close?.();
    map.delete(oldestKey);
  }
}

function cacheThumb(key, bitmap) {
  touch(thumbCache, key, bitmap);
  trim(thumbCache, THUMB_CACHE_LIMIT);
}

async function fallbackDecode(task) {
  let source = sourceCache.get(task.uri);
  if (source) {
    touch(sourceCache, task.uri, source);
  } else {
    source = await getBitmap(task.uri);
    touch(sourceCache, task.uri, source);
    trim(sourceCache, SOURCE_CACHE_LIMIT);
  }

  const sourceWidth = source.naturalWidth || source.width;
  const sourceHeight = source.naturalHeight || source.height;
  const [x1Raw, y1Raw, x2Raw, y2Raw] = task.bbox;
  const x1 = Math.max(0, Math.min(sourceWidth - 1, Number(x1Raw) || 0));
  const y1 = Math.max(0, Math.min(sourceHeight - 1, Number(y1Raw) || 0));
  const x2 = Math.max(x1 + 1, Math.min(sourceWidth, Number(x2Raw) || sourceWidth));
  const y2 = Math.max(y1 + 1, Math.min(sourceHeight, Number(y2Raw) || sourceHeight));
  const cropW = Math.max(1, x2 - x1);
  const cropH = Math.max(1, y2 - y1);
  const scale = task.size / Math.max(cropW, cropH);
  const drawW = cropW * scale;
  const drawH = cropH * scale;
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(task.size, task.size)
    : document.createElement('canvas');
  canvas.width = task.size;
  canvas.height = task.size;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, task.size, task.size);
  ctx.drawImage(source, x1, y1, cropW, cropH, (task.size - drawW) / 2, (task.size - drawH) / 2, drawW, drawH);
  if (typeof canvas.transferToImageBitmap === 'function') return canvas.transferToImageBitmap();
  return await createImageBitmap(canvas);
}

function sortQueue() {
  queued.sort((a, b) => a.priority - b.priority || a.seq - b.seq);
}

function pumpFallback() {
  while (activeFallback < WORKER_COUNT && queued.length > 0) {
    sortQueue();
    const task = queued.shift();
    if (!task || task.cancelled) continue;
    activeFallback += 1;
    fallbackDecode(task)
      .then((bitmap) => {
        cacheThumb(task.cacheKey, bitmap);
        counters.completed += 1;
        if (!task.cancelled) task.onBitmap?.(bitmap);
      })
      .catch((error) => {
        counters.failed += 1;
        if (!task.cancelled) task.onError?.(error);
      })
      .finally(() => {
        activeFallback -= 1;
        pumpFallback();
      });
  }
}

function runFallbackTask(task) {
  activeFallback += 1;
  fallbackDecode(task)
    .then((bitmap) => {
      cacheThumb(task.cacheKey, bitmap);
      counters.completed += 1;
      if (!task.cancelled) task.onBitmap?.(bitmap);
    })
    .catch((error) => {
      counters.failed += 1;
      if (!task.cancelled) task.onError?.(error);
    })
    .finally(() => {
      activeFallback -= 1;
      pumpWorkers();
    });
}

function getIdleWorker() {
  return workers.find((workerState) => !workerState.task);
}

function pumpWorkers() {
  if (!canUseWorkers || workers.length === 0) {
    pumpFallback();
    return;
  }

  let idle = getIdleWorker();
  while (idle && queued.length > 0) {
    sortQueue();
    const task = queued.shift();
    if (!task || task.cancelled) continue;
    idle.task = task;
    inflight.set(task.id, idle);
    idle.worker.postMessage({ id: task.id, uri: task.uri, bbox: task.bbox, size: task.size });
    idle = getIdleWorker();
  }
}

function watchQueuePressure() {
  if (queued.length <= 200) {
    if (queueWarningTimer) window.clearTimeout(queueWarningTimer);
    queueWarningTimer = null;
    return;
  }
  if (queueWarningTimer) return;
  queueWarningTimer = window.setTimeout(() => {
    queueWarningTimer = null;
    if (queued.length > 200) {
      console.warn(`Image loader queue is high (${queued.length} pending tasks).`);
    }
  }, 2000);
}

function initWorkers() {
  if (!canUseWorkers || workers.length > 0) return;
  for (let i = 0; i < WORKER_COUNT; i += 1) {
    const worker = new Worker(new URL('./decodeWorker.js', import.meta.url), { type: 'module' });
    const workerState = { worker, task: null };
    worker.onmessage = ({ data }) => {
      const task = workerState.task;
      workerState.task = null;
      if (task) {
        inflight.delete(task.id);
        if (data?.bitmap) {
          cacheThumb(task.cacheKey, data.bitmap);
          counters.completed += 1;
          if (!task.cancelled) task.onBitmap?.(data.bitmap);
        } else {
          runFallbackTask(task);
        }
      }
      pumpWorkers();
    };
    worker.onerror = (error) => {
      const task = workerState.task;
      workerState.task = null;
      if (task) {
        inflight.delete(task.id);
        queued.push(task);
      }
      counters.failed += 1;
      pumpFallback();
      error.preventDefault?.();
    };
    workers.push(workerState);
  }
}

export const imageLoaderPool = {
  enqueue({ uri, bbox, size = 120, priority = 1, onBitmap, onError }) {
    const normalized = {
      id: nextId,
      seq: nextId,
      uri,
      bbox: Array.isArray(bbox) && bbox.length === 4 ? bbox : [0, 0, 0, 0],
      size,
      priority: Number(priority),
      onBitmap,
      onError,
      cancelled: false,
    };
    nextId += 1;
    normalized.cacheKey = keyFor(normalized);

    const cached = thumbCache.get(normalized.cacheKey);
    if (cached) {
      counters.cacheHit += 1;
      touch(thumbCache, normalized.cacheKey, cached);
      queueMicrotask(() => {
        if (!normalized.cancelled) normalized.onBitmap?.(cached);
      });
    } else {
      counters.cacheMiss += 1;
      queued.push(normalized);
      watchQueuePressure();
      initWorkers();
      pumpWorkers();
    }

    return {
      cancel() {
        normalized.cancelled = true;
        const index = queued.findIndex((task) => task.id === normalized.id);
        if (index >= 0) queued.splice(index, 1);
      },
      boost(newPriority) {
        normalized.priority = Number(newPriority);
        sortQueue();
        pumpWorkers();
      },
    };
  },

  clearCache() {
    for (const bitmap of thumbCache.values()) bitmap?.close?.();
    thumbCache.clear();
    sourceCache.clear();
  },

  stats() {
    return {
      queued: queued.length,
      inflight: inflight.size + activeFallback,
      cacheHit: counters.cacheHit,
      cacheMiss: counters.cacheMiss,
      completed: counters.completed,
      failed: counters.failed,
      memoryMB: Math.round((thumbCache.size * 120 * 120 * 4) / 1024 / 1024),
      workers: canUseWorkers ? workers.length : 0,
    };
  },
};
