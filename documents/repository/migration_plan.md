# Repository Migration Plan

## Target Structure

```text
tools/
  download_models.py
  ...

object_detection/
  dino/
  grounding_dino/
  yolo/

segmentation/
  unet/
  sam/
    no_finetune/
    finetune/
    backbones/
    shared/

ui/
  editor_app/
```

## Migration Strategy

This migration is intentionally **non-destructive** in phase 1:

- Keep the legacy top-level packages (`dino`, `sam`, `sam_finetune`, `unet`, `editor_app`) working.
- Add the new namespace packages as **wrappers** that forward into the legacy implementations.
- Move notebooks, docs, and shell commands over to the new paths first.
- Only after the new paths are stable should internal source files be moved physically.

## New Command Paths

### Object Detection

- Old: `python -m dino`
- New: `python -m object_detection.dino`

- Old: `python tools/gdino_detect_image.py`
- New: `python -m object_detection.grounding_dino.image`

- Old: `python tools/gdino_detect_folder.py`
- New: `python -m object_detection.grounding_dino.folder`

### Segmentation

- Old: `python -m unet`
- New: `python -m segmentation.unet`

- Old: `python unet/train.py`
- New: `python -m segmentation.unet.train`

- Old: `python -m sam`
- New: `python -m segmentation.sam.no_finetune`

- Old: `python -m sam_finetune`
- New: `python -m segmentation.sam.finetune`

- Old: `python -m sam_finetune.train`
- New: `python -m segmentation.sam.finetune.train`

- Old: `python -m sam_finetune.test`
- New: `python -m segmentation.sam.finetune.test`

- Old: `python -m sam_finetune.pseudo_label`
- New: `python -m segmentation.sam.finetune.pseudo_label`

### UI

- Old: `python -m editor_app`
- New: `python -m ui.editor_app`

## Packaging

`setup.py` now provides:

- installable console entry points for the new namespaces
- a `download_models` custom command
- a unified `tools/download_models.py` script for named model presets

Examples:

```bash
python setup.py download_models --name sam_vit_b
python -m tools.download_models list
python -m tools.download_models download --name grounding_dino_base
```

## Phase 1: Done

- Added `object_detection/`, `segmentation/`, and `ui/` namespaces.
- Added wrappers for DINO, GroundingDINO, UNet, SAM runtime, and SAM finetune.
- Added `setup.py`.
- Added `tools/download_models.py`.

## Current Status

- `segmentation/sam/backbones/segment_anything` is now the shared SAM base.
- `segmentation/sam/no_finetune` is the canonical pure-SAM runtime path.
- `segmentation/sam/finetune` keeps only finetune-specific layers such as delta tuning, trainers, and tiled inference.
- `segmentation/datasets/core` is shared between `unet` and `sam_finetune`.
