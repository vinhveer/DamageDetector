# Segment Tab: Folder Queue & Mask Cutout Export

## Summary

Added two major features to the Segment tab:

1. **Folder Queue Workflow** — Load a folder of images, annotate each one sequentially, and auto-advance through the queue
2. **Mask Cutout Export** — Export transparent PNG cutouts (keep masked pixels, transparent elsewhere) alongside overlay/mask

---

## 1. Mask Cutout Export

### Python Backend

**Files modified:**
- `segmentation/sam/point_predict.py`
- `segmentation/sam/text_predict.py`

**What it does:**
- After computing the best mask, creates an RGBA PNG where:
  - Masked pixels retain original RGB values with full opacity
  - Non-masked pixels are fully transparent
  - Image is cropped tight to the mask bounding box
- Saves as `{base}_point_sam_cutout.png` or `{base}_text_sam_cutout.png`
- Returns `cutout_path` and `cutout_b64` in the JSON result

**Output structure:**
```
results_point_sam/
  image_001_point_sam_overlay.png   # visualization with points/contours
  image_001_point_sam_mask.png      # binary mask (white=mask, black=background)
  image_001_point_sam_cutout.png    # NEW: transparent cutout (RGBA)
```

---

## 2. Folder Queue Workflow

### Redux State (`segmentSlice.js`)

**New state fields:**
```js
queue: [],           // array of image paths from selected folder
queueIndex: 0,       // current position in queue
processed: {},       // { [imagePath]: 'done' | 'error' }
autoAdvance: true,   // auto-jump to next image after successful run
```

**New actions:**
- `setQueue(paths)` — load folder images into queue
- `clearQueue()` — exit queue mode
- `setQueueIndex(idx)` — navigate to specific image (resets points/status/result)
- `markProcessed({ path, status })` — mark image as done/error
- `setAutoAdvance(bool)` — toggle auto-advance behavior

**Updated thunk:**
- `runSegmentation()` now:
  - Captures `cutout_path` and `cutout_b64` from result
  - Marks current image as processed (done/error)
  - Auto-advances to next image if `autoAdvance` is enabled and queue has more images

---

### UI (`SegmentTab.jsx`)

**Image picker section:**
- Added "Browse Folder" button (folder icon) next to "Browse Image"
- When folder is loaded, shows queue navigator panel with:
  - Current position: "Queue 3 / 15" + done/error counts
  - Prev/Next buttons (disabled at boundaries)
  - Current image name with status icon (✓ done, ⚠ error)
  - "Auto-advance to next after Run" checkbox
  - "Clear" button to exit queue mode

**Run button:**
- Label changes to "Run & Next" when queue is active and auto-advance is on

**Result panel:**
- Added "Cutout (mask only)" section showing:
  - Transparent PNG preview (checkerboard background)
  - File path below preview

**Empty state:**
- Added "Browse Folder" button alongside "Browse Image"

---

## Usage

### Single Image Mode (existing)
1. Click "Browse Image" or folder icon
2. Place points (or enter text prompt)
3. Click "Run Segmentation"
4. View overlay, mask, and cutout in result panel

### Folder Queue Mode (new)
1. Click "Browse Folder" (folders icon)
2. Select a directory — all images load into queue
3. Place points on first image
4. Click "Run & Next" — saves result, marks as done, jumps to next image
5. Repeat for each image
6. Navigate manually with Prev/Next if needed
7. Uncheck "Auto-advance" to stay on current image after run

**Queue indicators:**
- Green checkmark (✓) = processed successfully
- Red warning (⚠) = error during processing
- No icon = not yet processed

---

## Files Changed

### Python
- `segmentation/sam/point_predict.py` — added cutout generation
- `segmentation/sam/text_predict.py` — added cutout generation

### Renderer (React)
- `app/src/features/segment/segmentSlice.js` — queue state, actions, thunk updates
- `app/src/features/segment/SegmentTab.jsx` — folder browse, queue UI, cutout display

### No changes needed
- `app/electron/main.js` — already supports `listImageFiles` IPC
- `app/electron/preload.cjs` — already exposes `listImageFiles`

---

## Testing

```bash
cd DamageDetector/app
npm run dev
```

1. Open Segment tab
2. Click "Browse Folder", select a folder with multiple images
3. Place points on first image
4. Click "Run & Next" — should advance to next image automatically
5. Check output folder — should contain overlay, mask, and cutout PNGs
6. Verify cutout has transparent background (open in image viewer)
7. Navigate with Prev/Next buttons
8. Uncheck "Auto-advance", run again — should stay on current image
9. Click "Clear" to exit queue mode

---

## Notes

- Queue is in-memory only (not persisted across sessions)
- Auto-advance only triggers on successful runs (not on errors)
- Cutout is only generated when mask has non-zero area (bbox exists)
- Cutout is cropped to mask bounding box to save space
- Single-image mode still works as before (no queue UI shown)
