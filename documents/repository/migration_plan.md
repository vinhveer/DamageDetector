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
    runtime/
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
- New: `python -m segmentation.sam.runtime`

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

## Phase 2: Next

- Move shared internals from `sam/` and `sam_finetune/` into `segmentation/sam/backbones` and `segmentation/sam/shared`.
- Move `unet/` internals into `segmentation/unet/`.
- Move `editor_app/` internals into `ui/editor_app/`.
- Consolidate docs so examples prefer only the new commands.

## Phase 3: Cleanup

- Remove legacy top-level wrappers after notebooks and scripts stop using them.
- Update CI/tests to target only the new namespace packages.
