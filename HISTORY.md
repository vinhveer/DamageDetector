# History Feature Spec

## Scope
History tracks prediction runs and their per-image masks for a workspace. It enables:
- Persisted results across restarts
- Viewing masks from a specific run or for a specific image
- Batch (folder) and single-image runs stored consistently

## Terminology
- Workspace: a folder containing `images/` and `results/`.
- Run: one prediction execution (single image or batch). Each run has a `run_id`.
- Image history: masks produced for a single image across runs.
- Folder history: masks produced for all images in a run (batch or single).

## Storage Layout
All history data lives under the current workspace:

```
<workspace>/
  images/
  results/
    <image_name>/
      data.csv
      <run_id>__ket_qua_lan_quet_workspace__det_001.png
      <run_id>__ket_qua_lan_quet_workspace__det_002.png
      ...
    <run_id>_lan_quet_workspace.csv
```

### Per-image folder
`<image_name>/data.csv` stores all detections for that image across runs. Each detection is one row.

### Per-run file
`<run_id>_lan_quet_workspace.csv` stores all detections for that run across all images. Each detection is one row.

## CSV Schema
Both `data.csv` and `<run_id>_lan_quet_workspace.csv` share the same columns:

- `run_id`: unique identifier for the run
- `created_at`: ISO timestamp when the run started
- `model`: model name (e.g., `SamDino`, `UnetDino`)
- `label`: detection label
- `score`: detection score (float)
- `mask_rel`: path relative to `<workspace>/results/`
- `image_rel`: path relative to `<workspace>/` (or absolute if outside workspace)
- `box`: optional `x1,y1,x2,y2`

## Run ID
- Format: `YYYYMMDD_HHMMSS` (with suffix `_02`, `_03` if collision)
- One run is created per prediction invocation (single image or batch).

## UI Behavior
### Folder History (Toolbar)
- Opens a dialog to select a run (`*_lan_quet_workspace.csv`).
- Then prompts to select an image from that run.
- Loads that image and displays **all masks from the selected run**.

### Image History (Toolbar)
- Opens a dialog to select a run for the currently loaded image.
- If `All (no filter)` is selected, show all masks from all runs for that image.
- Otherwise show all masks from the selected run for that image.

## Loading Results on Startup
When opening a workspace:
- All `results/<image_name>/data.csv` files are loaded.
- UI mask list is populated for the currently opened image (if any).

## Persistence Keys (.settings)
- `workspace_path`: last opened workspace
- `last_folder_path`: last folder shown in Explorer

## Notes
- All masks are saved to disk on each detection.
- For batch runs, `results/<run_id>_lan_quet_workspace.csv` is always created.
- A single-image run also creates a `<run_id>_lan_quet_workspace.csv` with one image entry.
