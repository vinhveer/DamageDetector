# Prototype Review – Edit mode: tách luồng crop khỏi luồng app

Spec cho việc refactor thuật toán load + crop thumbnail trong tab **Prototype
Review** (Edit mode) tại
[`app/src/features/prototypeReview/PrototypeReview.jsx`](../app/src/features/prototypeReview/PrototypeReview.jsx).

Kiến trúc chốt: **dual-track** — Phase A (pre-generate trong Python pipeline)
là đường chính, Phase B (runtime sharp + raw-buffer worker pool) là fallback
khi DB chưa có crop hoặc file crop bị xóa.

---

## 1. Tại sao thuật toán hiện tại tệ

Luồng hiện tại (Edit mode):

```
IntersectionObserver
  → batch 80ms (useBatchedAssignmentLoader)
    → IPC listPrototypeReviewAssignmentsBulk
      → main process: 3 SQLite connection, 3 query sync (better-sqlite3)
        → trả về tất cả rows cho mọi cluster trong batch
          → React render thumbnails
            → mỗi <CroppedThumb> tự gọi getBitmap(image_uri)
              → fetch + createImageBitmap (FULL source image) trên render thread
                → drawImage crop vùng nhỏ vào canvas 120×120
```

**Vấn đề liệt kê:**

| # | Vấn đề | Hậu quả |
|---|--------|---------|
| 1 | `listPrototypeReviewAssignmentsBulk` chạy 3 SQLite query sync trong Electron main process | Main process block vài trăm ms → toàn app đứng khi user scroll nhanh |
| 2 | Mỗi thumbnail decode **full source image** (có khi 4000×3000) chỉ để cắt vùng 120×120 | CPU + RAM cao, GC nặng, scroll giật |
| 3 | LRU cache bitmap chỉ giữ 50 entries ([imageCache.js:2](../app/src/utils/imageCache.js)) | Cluster có ảnh từ nhiều source khác nhau → liên tục evict + decode lại |
| 4 | Không có concurrency cap | Một row có 50 box → fire 50 `fetch` + `createImageBitmap` đồng thời trên render thread |
| 5 | Crop làm trên render thread (`drawImage` trong `useEffect`) | Chặn paint, frame drop |
| 6 | Không streaming: cả batch chờ IPC trả về xong mới hiện toàn bộ thumbnail | User thấy "Loading…" dài, rồi nhảy ra một đống → không có cảm giác "đang chạy" |
| 7 | Không có layout reservation | Khi thumbnail nhảy vào, content shift, scrollbar nhảy |
| 8 | Không virtualize | List 500+ cluster → 500 DOM rows mount cùng lúc |
| 9 | Mỗi lần mở lại edit mode lại decode lại từ đầu | Không persist, không tận dụng work đã làm |

Với hàng trăm box / cluster và list cả nghìn cluster, không cách on-demand
nào ở renderer side đủ nhanh. Phải pre-generate.

---

## 2. Mục tiêu

1. **Luồng app (renderer + Electron main) tuyệt đối không cắt ảnh on the render path.**
2. **Crop generation chạy 1 lần** (Phase A — ở Python pipeline) hoặc **chạy
   trong worker_threads** (Phase B — fallback runtime), **không bao giờ** chạy
   trên render/main thread.
3. App chỉ làm `<img src="file://.../crop.jpg">` → browser tự decode trên
   image decoder thread riêng.
4. Hàng trăm box/cluster, nghìn cluster vẫn mượt 60fps.
5. Re-open edit mode = 0ms latency (tất cả crops đã có sẵn trên đĩa).

---

## 3. Kiến trúc dual-track

```
                            ┌──── App (Electron + React) ────┐
                            │                                │
                            │  <img src="file://.../<x>.jpg">│
                            │  ← chỉ là image tag thuần      │
                            └─────────────▲──────────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │  Resolve crop file path:      │
                          │  1. DB có `thumb_path`?       │
                          │     → dùng → DONE             │
                          │  2. Cache đĩa <hash>.jpg có?  │
                          │     → dùng → DONE             │
                          │  3. Fallback: enqueue Phase B │
                          └───────┬───────────────────────┘
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       │ PHASE A (chính, pre-gen)    PHASE B (fallback)      │
       │ ─────────────────────       ─────────────────────   │
       │ Python pipeline:            Electron worker_threads │
       │ • step 2 đã có crop_path    • sharp + raw buffer    │
       │ • Thêm thumbnail_path 256px │ • group by image_path │
       │ • Lưu JPEG q80 ra disk      │ • write <sha1>.jpg    │
       │ • SQLite cột mới: thumb_path│ • emit crops:ready    │
       │ • Chạy 1 lần khi gen DB     │ • cache userData/crops│
       └──────────────────────────────────────────────────────┘
```

**Quy tắc fallback:**
- DB có `thumb_path` non-empty + file tồn tại → Phase A win, skip Phase B.
- DB không có cột / cột rỗng / file đã bị xóa → Phase B chạy on-demand, cache
  vào `userData/crops/`.
- Một khi Phase B đã cache, lần mở sau không trigger lại.

---

## 4. PHASE A — Pre-generate trong Python pipeline (đường chính)

### 4.1. Tình trạng hiện tại

Code Python đã có sẵn:
- [`semi-labeling/step2_sematic/pipeline.py`](../semi-labeling/step2_sematic/pipeline.py)
  có flag `--save-crops` + `--crop-dir`
- Default save: `{db_parent}/step2_sematic_crops/<semantic_run_id>/<image_rel_path_parent>/<stem>__det<detection_id>.png`
- Schema [`sqlite_store.py:75`](../semi-labeling/step2_sematic/sqlite_store.py#L75):
  cột `crop_path TEXT NOT NULL` trong `openclip_semantic_results`

Nhưng:
- Lưu **PNG lossless** → 10–20× bigger than JPEG, cluster 100 box × nghìn ảnh =
  GB dung lượng vô ích cho thumbnail.
- Lưu ở **kích thước gốc box** (1000×1500 với box to) → vẫn phải decode lại
  để hiển thị thumbnail nhỏ.
- `--save-crops` mặc định `False` → user phải nhớ bật → đa số DB hiện tại
  không có.

### 4.2. Thay đổi đề xuất

#### 4.2.1. Schema mở rộng

Thêm cột mới (không phá schema cũ — `crop_path` giữ nguyên cho crops gốc khi
cần debug OpenCLIP):

```sql
ALTER TABLE openclip_semantic_results ADD COLUMN thumb_path TEXT NOT NULL DEFAULT '';
-- Đường dẫn (absolute hoặc relative tới db_parent) tới file thumbnail JPEG nhỏ,
-- max-side = THUMB_MAX_SIDE px. Empty string = chưa generate.
```

Migration: code đọc DB phải tolerant với cột thiếu — wrap trong `try/except`
hoặc `PRAGMA table_info` check.

#### 4.2.2. Thumbnail generation luôn bật

Trong `Step2SemanticPipeline.run()`:

- **Bỏ flag `--save-crops`** (hoặc giữ flag nhưng đổi nghĩa: tách thumbnail
  vs full-resolution crop):
  - Thumbnail (luôn lưu, dùng cho UI): JPEG quality 80, max-side 256px
  - Full crop (chỉ khi `--save-full-crops`, dùng debug): PNG lossless như cũ
- Path layout mới:
  ```
  {db_parent}/thumbs/<semantic_run_id>/<image_rel_path_parent>/<stem>__det<id>.jpg
  ```
- Khi crop image cho OpenCLIP (đã decode rồi), pipe tiếp:
  ```python
  thumb = crop_image.copy()
  thumb.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
  thumb.save(thumb_path, format="JPEG", quality=80, optimize=True)
  ```
- `OpenClipSemanticClassifier` đã decode PIL image → free thumbnail.

Const đề xuất:
- `THUMB_MAX_SIDE = 256` — vừa cho UI 120×120 retina (240px) + dự phòng zoom.
- Format: JPEG q80 progressive — nhỏ, fast decode trong browser.

#### 4.2.3. Lưu `thumb_path` vào DB

Trong `_persist_result` hoặc tương đương:
```python
store.insert_result(
    ...,
    crop_path=str(full_crop_path) if save_full else "",
    thumb_path=str(thumb_path),
    ...
)
```

Lưu **path tuyệt đối** thay vì relative (tránh ambiguity khi DB bị di chuyển).
App resolve `thumb_path`:
- Nếu absolute + exists → dùng.
- Nếu absolute nhưng không exists → coi như missing, fallback Phase B.
- Nếu relative → resolve tương đối db_parent.

#### 4.2.4. Standalone backfill script

Cho user có DB cũ (chưa có thumb): script standalone
`semi-labeling/step2_sematic/backfill_thumbs.py`:

```bash
python semi-labeling/step2_sematic/backfill_thumbs.py \
  --db /path/to/damage_scan.sqlite3 \
  --image-root /path/to/HinhAnh \
  --thumb-max-side 256 \
  --quality 80 \
  --workers 8
```

Logic:
- Query mọi row có `thumb_path = ''` hoặc cột chưa tồn tại
- Group by `image_path` (key insight — để decode source 1 lần)
- Multiprocess pool (`multiprocessing.Pool` size = CPU - 1), mỗi worker:
  - Decode source 1 lần qua PIL
  - Crop tất cả box thuộc source đó
  - Resize + save JPEG q80
  - Trả về list `(result_id, thumb_path)`
- Main: batch UPDATE SQLite

Progress bar (tqdm) để user thấy backfill bao xa.

### 4.3. Đường dẫn join để app dùng được

`feature_group_assignments` (step 4) chứa `result_id` → join về
`openclip_semantic_results.thumb_path`:

```sql
SELECT
  fga.result_id, fga.cluster_key, fga.distance_to_center, ...,
  osr.thumb_path, osr.image_path, osr.x1, osr.y1, osr.x2, osr.y2
FROM feature_group_assignments AS fga
JOIN openclip_semantic_results AS osr
  ON osr.result_id = fga.result_id
WHERE fga.grouping_run_id = ? AND fga.cluster_key IN (...)
ORDER BY ...
```

Sửa [`app/electron/prototype_review/index.js`](../app/electron/prototype_review/index.js)
`sourceMeta()` và `listAssignmentsFromFeatureDb()` để thêm `thumb_path` vào
SELECT.

### 4.4. Dung lượng dự kiến

| Param | Giá trị |
|-------|---------|
| Avg thumbnail JPEG q80 256px | ~12 KB |
| 1 cluster × 100 box | ~1.2 MB |
| 500 cluster | ~600 MB |
| 2000 cluster | ~2.4 GB |

Chấp nhận được. Nếu user concern, có option giảm `THUMB_MAX_SIDE` xuống 192
(~7KB/file).

---

## 5. PHASE B — Runtime fallback (worker_threads + sharp)

### 5.1. Khi nào kích hoạt

App gọi `resolveThumb(row)` trong main process trước khi trả assignments
cho renderer:

```js
const resolveThumb = (row) => {
  // 1. DB có thumb_path?
  if (row.thumb_path && fs.existsSync(row.thumb_path)) {
    return { source: 'phase_a', fileUrl: pathToFileURL(row.thumb_path).href };
  }
  // 2. Cache đĩa Phase B đã có?
  const key = hashKey(row);
  const cached = path.join(CROP_CACHE_DIR, `${key}.jpg`);
  if (fs.existsSync(cached)) {
    return { source: 'phase_b_cached', fileUrl: pathToFileURL(cached).href };
  }
  // 3. Cần Phase B generate
  return { source: 'phase_b_pending', jobSpec: { key, ...cropSpec(row) } };
};
```

Row trả về có thêm field `thumb` = `{ fileUrl } | { pending: jobId }`.
Renderer dispatch crop job khi gặp `pending`.

### 5.2. IPC bổ sung

[`app/electron/preload.cjs`](../app/electron/preload.cjs):

```js
requestCrops: (jobs)    => ipcRenderer.invoke('crops:request', { jobs }),
cancelCrops: (jobIds)   => ipcRenderer.invoke('crops:cancel', { jobIds }),
onCropReady: (callback) => {
  const listener = (_e, payload) => callback(payload);
  ipcRenderer.on('crops:ready', listener);
  return () => ipcRenderer.removeListener('crops:ready', listener);
}
```

[`app/electron/main.js`](../app/electron/main.js):

```js
ipcMain.handle('crops:request', (_e, { jobs }) => cropQueue.enqueue(jobs));
ipcMain.handle('crops:cancel',  (_e, { jobIds }) => cropQueue.cancel(jobIds));
```

### 5.3. Worker pool — `sharp` raw-buffer trick

Tại sao raw-buffer? Hàng trăm box/cluster, đa số 50–100 box đến từ 10–30 ảnh
gốc → group jobs theo `imagePath`, decode source RGB raw **1 lần**, sau đó
mỗi extract chỉ là memcpy.

File `app/electron/crop_pipeline/index.js`:

```js
import { Worker } from 'node:worker_threads';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import fs from 'node:fs';
import { app, BrowserWindow } from 'electron';

const CACHE_DIR = path.join(app.getPath('userData'), 'crops');
const POOL_SIZE = Math.max(2, Math.min(6, os.cpus().length - 1));
const MAX_QUEUE = 4000;

class CropQueue {
  constructor() {
    this.queue = [];         // [{ jobId, key, imagePath, x1,y1,x2,y2, size }]
    this.workers = [];       // [{ worker, busy: false, currentImagePath: null }]
    this.spawnWorkers();
    fs.mkdirSync(CACHE_DIR, { recursive: true });
  }

  spawnWorkers() {
    for (let i = 0; i < POOL_SIZE; i++) {
      const w = new Worker(path.join(__dirname, 'crop_worker.js'));
      w.on('message', (msg) => this.onDone(i, msg));
      w.on('error', (err) => console.error('crop worker error', err));
      this.workers.push({ worker: w, busy: false, currentImagePath: null });
    }
  }

  enqueue(jobs) {
    const accepted = [];
    for (const job of jobs) {
      if (this.queue.length >= MAX_QUEUE) break;
      job.key = this.hashKey(job);
      const cached = path.join(CACHE_DIR, `${job.key}.jpg`);
      if (fs.existsSync(cached)) {
        process.nextTick(() => this.emit(job, cached));
        accepted.push(job.jobId);
        continue;
      }
      this.queue.push(job);
      accepted.push(job.jobId);
    }
    this.drain();
    return { accepted };
  }

  cancel(jobIds) {
    const ids = new Set(jobIds);
    this.queue = this.queue.filter(j => !ids.has(j.jobId));
  }

  drain() {
    // Greedy: mỗi worker grab tất cả jobs cùng imagePath với job đầu queue
    for (let i = 0; i < this.workers.length; i++) {
      if (this.workers[i].busy || this.queue.length === 0) continue;
      const head = this.queue[0];
      // pull tất cả jobs cùng imagePath ra batch
      const batch = [];
      const remaining = [];
      for (const j of this.queue) {
        if (j.imagePath === head.imagePath && batch.length < 64) batch.push(j);
        else remaining.push(j);
      }
      this.queue = remaining;
      this.workers[i].busy = true;
      this.workers[i].currentImagePath = head.imagePath;
      this.workers[i].worker.postMessage({
        batch,
        cacheDir: CACHE_DIR
      });
    }
  }

  onDone(workerIndex, { results }) {
    this.workers[workerIndex].busy = false;
    this.workers[workerIndex].currentImagePath = null;
    for (const r of results) {
      if (r.error) continue;
      const file = path.join(CACHE_DIR, `${r.key}.jpg`);
      this.emit({ jobId: r.jobId, key: r.key }, file);
    }
    this.drain();
  }

  emit(job, file) {
    const url = `file://${file}`;
    for (const win of BrowserWindow.getAllWindows()) {
      win.webContents.send('crops:ready', { jobId: job.jobId, fileUrl: url });
    }
  }

  hashKey({ imagePath, x1, y1, x2, y2, size }) {
    const data = `${imagePath}|${Math.round(x1)},${Math.round(y1)},${Math.round(x2)},${Math.round(y2)}|${size}`;
    return crypto.createHash('sha1').update(data).digest('hex');
  }
}

export const cropQueue = new CropQueue();
```

File `app/electron/crop_pipeline/crop_worker.js`:

```js
import { parentPort } from 'node:worker_threads';
import sharp from 'sharp';
import path from 'node:path';

sharp.cache({ memory: 512, items: 100, files: 0 });
sharp.concurrency(1);  // mỗi worker single-thread, pool ngoài đã song song

parentPort.on('message', async ({ batch, cacheDir }) => {
  const results = [];
  if (!batch.length) { parentPort.postMessage({ results }); return; }

  const imagePath = batch[0].imagePath;

  try {
    // KEY TRICK: decode source 1 lần thành raw RGB buffer
    const { data, info } = await sharp(imagePath, { failOn: 'none' })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const W = info.width;
    const H = info.height;

    for (const job of batch) {
      try {
        const left   = Math.max(0, Math.min(W - 1, Math.round(job.x1)));
        const top    = Math.max(0, Math.min(H - 1, Math.round(job.y1)));
        const width  = Math.max(1, Math.min(W - left, Math.round(job.x2 - job.x1)));
        const height = Math.max(1, Math.min(H - top,  Math.round(job.y2 - job.y1)));

        const out = path.join(cacheDir, `${job.key}.jpg`);
        await sharp(data, { raw: { width: W, height: H, channels: 3 } })
          .extract({ left, top, width, height })
          .resize(job.size, job.size, { fit: 'inside', withoutEnlargement: false })
          .jpeg({ quality: 78, mozjpeg: true, progressive: false })
          .toFile(out);
        results.push({ jobId: job.jobId, key: job.key });
      } catch (e) {
        results.push({ jobId: job.jobId, key: job.key, error: String(e.message || e) });
      }
    }
  } catch (e) {
    for (const job of batch) {
      results.push({ jobId: job.jobId, key: job.key, error: `decode-source: ${e.message || e}` });
    }
  }

  parentPort.postMessage({ results });
});
```

### 5.4. Vì sao Phase B vẫn cần dù có Phase A

1. **DB cũ chưa regenerate**: user chưa run backfill script → vẫn dùng app
   được, chỉ là crop lần đầu tốn ~50ms/box.
2. **File bị xóa thủ công** hoặc đổi `image_root`: `thumb_path` trong DB chỉ
   về file đã mất.
3. **Phase A backfill chưa xong**: trong lúc backfill chạy ngầm, user có thể
   mở edit mode → Phase B làm việc tạm.
4. **Box size khác `THUMB_MAX_SIDE`**: nếu UI sau này muốn thumb 384px /
   512px, Phase A 256px không đủ → Phase B sinh ra size mới on-demand.
5. **Test/preview workflows mới**: chạy SAM+GDino wizard ra detection box mới
   → chưa qua step 2 → chưa có thumb → Phase B handle.

### 5.5. Quota cache + cleanup

- Vị trí: `app.getPath('userData') + '/crops/'`
  (macOS: `~/Library/Application Support/Damage Detector/crops/`).
- Tên file: `<sha1>.jpg`.
- Cleanup chính sách:
  - Quota mặc định 2 GB. Khi vượt → xóa LRU theo `atime` xuống còn 1.5 GB.
  - Check quota lazy (mỗi 100 jobs hoàn thành), không cản đường drain.
  - Settings tab thêm:
    - "Clear runtime crop cache" button
    - Display "Crop cache: 487 MB / 2 GB"

---

## 6. Renderer-side

### 6.1. Hook mới — `useCropStream`

`app/src/features/prototypeReview/hooks/useCropStream.js`:

```js
import { useCallback, useEffect, useRef, useState } from 'react';

export function useCropStream() {
  const [cropByJobId, setCropByJobId] = useState({});  // jobId → fileUrl
  const requested = useRef(new Set());

  useEffect(() => window.electronAPI.onCropReady(({ jobId, fileUrl }) => {
    setCropByJobId((prev) => prev[jobId] === fileUrl ? prev : { ...prev, [jobId]: fileUrl });
  }), []);

  const requestCrop = useCallback((jobId, spec) => {
    if (requested.current.has(jobId)) return;
    requested.current.add(jobId);
    window.electronAPI.requestCrops([{ jobId, ...spec }]);
  }, []);

  const requestCropsBulk = useCallback((items) => {
    const fresh = items.filter(({ jobId }) => !requested.current.has(jobId));
    if (!fresh.length) return;
    fresh.forEach(({ jobId }) => requested.current.add(jobId));
    window.electronAPI.requestCrops(fresh);
  }, []);

  const cancelCrops = useCallback((jobIds) => {
    window.electronAPI.cancelCrops(jobIds);
    jobIds.forEach((id) => requested.current.delete(id));
  }, []);

  return { cropByJobId, requestCrop, requestCropsBulk, cancelCrops };
}
```

### 6.2. CroppedThumb mới — zero canvas

```jsx
function CroppedThumb({ row, size = 120, selected, onToggle, cropByJobId, requestCrop }) {
  // 1. Phase A win: thumb_path có sẵn
  if (row.thumb?.fileUrl) {
    return <ThumbButton fileUrl={row.thumb.fileUrl} {...{ row, size, selected, onToggle }} />;
  }

  // 2. Phase B: cần generate
  const jobId = row.thumb?.pending;
  const spec  = row.thumb?.spec;
  const fileUrl = jobId ? cropByJobId[jobId] : null;

  useEffect(() => {
    if (jobId && !fileUrl) requestCrop(jobId, spec);
  }, [jobId, fileUrl, spec, requestCrop]);

  return (
    <ThumbButton
      fileUrl={fileUrl}     // null khi chưa sẵn sàng
      placeholder
      {...{ row, size, selected, onToggle }}
    />
  );
}

function ThumbButton({ fileUrl, placeholder, row, size, selected, onToggle }) {
  return (
    <button
      type="button"
      onClick={() => onToggle(Number(row.result_id))}
      style={{ width: size, height: size }}
      className={cn(
        'relative flex-shrink-0 overflow-hidden rounded-[4px] border-2',
        selected ? 'border-[var(--primary)]' : 'border-[var(--border-muted)]'
      )}
    >
      {fileUrl
        ? <img src={fileUrl} width={size} height={size} loading="lazy" decoding="async"
               style={{ objectFit: 'contain', width: '100%', height: '100%' }} />
        : <div className="h-full w-full animate-pulse bg-[var(--surface-2)]" />
      }
      {selected && <CheckBadge />}
    </button>
  );
}
```

**Hai điểm:**
- Không còn `<canvas>` + `drawImage` → không tốn render thread.
- `<img src="file://...">` → browser decode trên thread riêng, lazy-load khi
  scroll vào view.

### 6.3. Per-cluster fetch + concurrency cap

Bỏ `useBatchedAssignmentLoader` (batch 80ms bulk). Thay bằng:

`app/src/features/prototypeReview/hooks/useClusterAssignments.js`:

```js
const INFLIGHT_CAP = 4;
const queue = [];
let inflight = 0;

function drain() {
  while (inflight < INFLIGHT_CAP && queue.length) {
    const next = queue.shift();
    inflight++;
    next.run().finally(() => { inflight--; drain(); });
  }
}

export function useClusterAssignments(selectedRun, paths, clusterKey, enabled) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const aborted = useRef(false);

  useEffect(() => {
    if (!enabled || !selectedRun || !clusterKey) return;
    aborted.current = false;
    setLoading(true);
    const task = {
      run: () => window.electronAPI.listPrototypeReviewAssignments({
        reviewDbPath: paths.reviewDbPath,
        reviewRunId: selectedRun.review_run_id,
        featureDbPath: selectedRun.feature_db_path,
        sourceDbPath: selectedRun.source_db_path,
        imageRootPath: paths.imageRootPath,
        groupingRunId: selectedRun.grouping_run_id,
        clusterKey
      }).then((result) => {
        if (aborted.current) return;
        setData(result.assignments || []);
      }).catch(() => {
        if (!aborted.current) setData([]);
      }).finally(() => {
        if (!aborted.current) setLoading(false);
      })
    };
    queue.push(task);
    drain();
    return () => { aborted.current = true; };
  }, [enabled, selectedRun?.review_run_id, clusterKey]);

  return { assignments: data, loading };
}
```

- Per-cluster query thay vì bulk → cancel dễ, IPC ngắn.
- Global semaphore 4 → không storm main process khi scroll nhanh.
- Abort khi component unmount.

### 6.4. Virtualization

Thêm `react-window`. Thay `editScores.map(...)` bằng `<FixedSizeList>`:

```jsx
import { FixedSizeList as List } from 'react-window';

<List
  height={containerHeight}        // dynamic, đo qua AutoSizer
  itemCount={editScores.length}
  itemSize={152}                  // header(32) + thumb-strip(120)
  overscanCount={3}
  itemData={{ editScores, selectedIds, deletedIds, onToggleImage, onToggleGroup }}
>
  {ClusterRow}
</List>
```

Mỗi `ClusterRow` style:
```css
height: 152px;
contain: layout paint style;
content-visibility: auto;
contain-intrinsic-size: 152px;
```

→ off-screen rows browser skip paint hoàn toàn.

### 6.5. Cancel khi scroll qua

`ClusterRow` cleanup `useEffect`:

```js
useEffect(() => () => {
  // 1. Cancel pending assignment fetch (aborted ref đã handle ở 6.3)
  // 2. Cancel pending crop jobs cho cluster này
  if (cluster?.pendingJobIds?.length) {
    cancelCrops(cluster.pendingJobIds);
  }
}, []);
```

User scroll cực nhanh từ đầu → cuối list: chỉ end-state cluster thật sự được
fetch + crop, các cluster giữa bị huỷ sạch.

---

## 7. Files cần đụng

### Tạo mới

**Python (Phase A):**
- `semi-labeling/step2_sematic/backfill_thumbs.py` — script backfill DB cũ
- (Tùy chọn) `semi-labeling/step2_sematic/thumb_utils.py` — helper resize+save

**Electron (Phase B):**
- `app/electron/crop_pipeline/index.js` — pool + queue
- `app/electron/crop_pipeline/crop_worker.js` — sharp raw-buffer worker

**Renderer:**
- `app/src/features/prototypeReview/hooks/useCropStream.js`
- `app/src/features/prototypeReview/hooks/useClusterAssignments.js`
- `app/src/features/prototypeReview/components/ClusterRow.jsx` (tách từ
  `ClusterEditRow`)

### Sửa

**Python:**
- [`semi-labeling/step2_sematic/pipeline.py`](../semi-labeling/step2_sematic/pipeline.py):
  - Luôn sinh thumbnail (bỏ phụ thuộc `--save-crops` cho thumbnail; flag cũ
    giữ riêng cho full crops)
  - Path layout mới `thumbs/<run_id>/...`
  - Quality JPEG q80, max-side 256
- [`semi-labeling/step2_sematic/sqlite_store.py`](../semi-labeling/step2_sematic/sqlite_store.py):
  - Schema thêm cột `thumb_path`
  - `insert_result()` nhận thêm param `thumb_path`

**Electron:**
- [`app/electron/main.js`](../app/electron/main.js) — wire IPC `crops:request`,
  `crops:cancel`
- [`app/electron/preload.cjs`](../app/electron/preload.cjs) — expose
  `requestCrops`, `cancelCrops`, `onCropReady`
- [`app/electron/prototype_review/index.js`](../app/electron/prototype_review/index.js):
  - SELECT thêm `thumb_path`
  - Trong `hydrateAssignments`: gọi `resolveThumb()` cho mỗi row, attach
    field `thumb`

**Renderer:**
- [`app/src/features/prototypeReview/PrototypeReview.jsx`](../app/src/features/prototypeReview/PrototypeReview.jsx):
  - Drop `useBatchedAssignmentLoader`
  - Drop canvas-based `CroppedThumb`
  - Add virtualization
  - Add `useCropStream`

**Package & docs:**
- [`app/package.json`](../app/package.json) — add `sharp`, `react-window`
- [`app/CLAUDE.md`](../app/CLAUDE.md) — pattern crop pipeline mới
- [`DamageDetector/CLAUDE.md`](../CLAUDE.md) — note thumbs dir mới

---

## 8. Migration plan (thứ tự commit, mỗi bước test độc lập)

### Bước 1 — Phase A schema + pipeline change
- Schema ALTER TABLE `openclip_semantic_results` add `thumb_path`
- Sửa `pipeline.py` luôn sinh thumb 256px JPEG q80
- Test: run step 2 trên 1 thư mục nhỏ, verify thumb files + DB column

### Bước 2 — Phase A backfill script
- `backfill_thumbs.py` với multiprocessing pool, group by image_path
- Test: chạy trên DB hiện có (`infer_results/.../step2_sematic/damage_scan.sqlite3`),
  verify mọi row có thumb_path + file thumb exists

### Bước 3 — Electron IPC + resolveThumb (chưa có Phase B)
- Sửa `prototype_review/index.js` SELECT thêm `thumb_path`
- Thêm `resolveThumb()` trả về `{ source, fileUrl }` cho row có Phase A,
  `{ source: 'phase_b_pending', jobSpec }` cho row chưa có (tạm trả error)
- Test: edit mode hiện tại load assignments có `thumb.fileUrl` đúng

### Bước 4 — Renderer dùng `<img>` thay canvas
- Rewrite `CroppedThumb` dùng `<img>` (đọc `row.thumb.fileUrl`)
- Test: edit mode trên DB đã backfill xong, scroll nhanh không lag, no canvas

### Bước 5 — Phase B worker pool + IPC
- `npm install sharp` (prebuild có sẵn cho 3 OS)
- Tạo `crop_pipeline/`, wire IPC `crops:request`, `crops:cancel`,
  event `crops:ready`
- Test: từ DevTools console, request 1 crop với image thật → file xuất hiện
  ở `userData/crops/`

### Bước 6 — useCropStream + fallback rendering
- Hook + dispatch crop khi row có `thumb.pending`
- Test: tạo 1 row giả với thumb_path empty, kiểm tra Phase B kick in, thumb
  hiện sau ~50ms

### Bước 7 — Per-cluster fetch + concurrency cap
- Loại `useBatchedAssignmentLoader`, thay bằng `useClusterAssignments`
  với INFLIGHT_CAP=4
- Test: scroll nhanh không phát sinh IPC storm (DevTools network/IPC)

### Bước 8 — Virtualize list
- Add `react-window` (~6KB gzip), `react-virtualized-auto-sizer`
- Replace `editScores.map` bằng `<List>`
- Test: 1000+ cluster scroll mượt ~60fps (Performance panel)

### Bước 9 — Cancel + cleanup
- Cancel pending crops khi `ClusterRow` unmount
- Quota check + LRU cleanup + Settings tab "Clear cache" button
- Test: scroll nhanh 1000 cluster, kiểm tra queue tự drain xuống, cache không
  vượt quota

### Bước 10 — Port pattern qua ResultViewer + DamageScan editor
- [`ImageGrid`](../app/src/features/resultViewer/components/ImageGrid.jsx) có
  cùng vấn đề, áp dụng cùng pattern (đọc thumb_path khi có)

---

## 9. Đo lường thành công

Đo trên DB thật (~500 cluster, ~50 box/cluster trung bình, source 3000×4000):

| Metric | Trước | Phase B fresh | Phase A win |
|--------|-------|---------------|-------------|
| Click "Edit" → thumb đầu tiên | 1500 ms | ~50 ms | ~5 ms |
| Block main process khi scroll | 300–800 ms / frame | < 16 ms | < 16 ms |
| RAM renderer (xem 500 cluster) | ~1.2 GB | ~200 MB | ~200 MB |
| CPU peak | ~95% (1 renderer core) | ~40% (worker pool) | ~5% |
| Frame rate scroll | 15–25 fps | ~55 fps | ~60 fps |
| 1 cluster × 100 box, lần đầu | 2–4 s + freeze | ~600 ms streaming | < 50 ms |
| Re-open cluster sau khi đã xem | re-decode lại | 0 ms (file cache) | 0 ms |

Cách đo: Chrome DevTools Performance panel, profile 5s scroll từ đầu đến giữa
list, compare main-thread blocking time.

---

## 10. Rủi ro & câu hỏi mở

1. **`sharp` native binding**: build phải có prebuild cho platform của user.
   Electron 42 + Node 20 → prebuild sẵn macOS arm64/x64, Linux x64,
   Win x64. Test trên cả 3.
2. **`file://` protocol + `webSecurity: false`**: đang bật ở
   [`electron/main.js:50`](../app/electron/main.js#L50) — `<img src="file://...">`
   load OK. Verify không break CSP nếu sau này bật.
3. **Worker khởi tạo lần đầu**: ~100ms × N workers. Khởi tạo lazy lúc nhận
   request đầu tiên, không spawn lúc app start.
4. **Phase A path absolute vs relative**: chọn absolute để app không phải
   resolve. Trade-off: DB không portable. Có thể thêm
   `--thumb-path-mode {absolute,relative}` flag.
5. **Schema migration cho DB cũ**: ALTER TABLE thêm cột `thumb_path` với
   `DEFAULT ''` → an toàn. App đọc DB phải tolerant với cột thiếu (PRAGMA
   table_info check, hoặc try/except SELECT).
6. **Race condition Phase A + Phase B chạy đồng thời cho cùng box**:
   tránh bằng cách: nếu DB có `thumb_path` non-empty → không enqueue Phase B
   dù file chưa tồn tại (assume backfill đang chạy, file sẽ xuất hiện).
   Refresh khi user click "Reload".
7. **Đường dẫn có ký tự đặc biệt (space, unicode, accent VN)**: dùng
   `pathToFileURL` của Node thay vì string concat `file://`.
8. **Box vượt biên ảnh**: clamp ở worker (đã có `Math.max(0, ...)`) + lấy
   `image_width/height` từ DB nếu muốn validate trước.
9. **Cluster có 0 box hoặc box invalid (x2 ≤ x1)**: worker trả error, hook
   render fallback `?` icon, không retry.
10. **Quota cache đụng đĩa user**: 2GB default. Settings UI để user override.
11. **Phase A thumb chất lượng JPEG q80 256px** có đủ tốt cho UI thumbnail
    120px display + 240px retina không? → q80 là chuẩn, đủ. Nếu sau này UI
    cần xem to hơn (vd lightbox 800px), dùng `crop_path` (full PNG) thay vì
    `thumb_path`.
12. **Phase A backfill cho DB rất lớn (10k+ row)**: multiprocessing pool 8
    worker dự kiến ~5–10 phút. OK.

---

## 11. Quyết định đã chốt

- **Option C — dual track**: Phase A pre-generate ưu tiên, Phase B fallback runtime
- Vị trí cache Phase B: `userData/crops/` (theo OS chuẩn)
- Dùng `sharp` (chấp nhận ~30MB native binding) cho tốc độ
- Quota cache 2GB default, LRU eviction
- Thumb format: JPEG q80 max-side 256px
- Schema: thêm cột `thumb_path` vào `openclip_semantic_results`, giữ
  `crop_path` cho full crops (debug)
