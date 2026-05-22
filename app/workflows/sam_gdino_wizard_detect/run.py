"""
SAM+GDino Wizard – Detection workflow
======================================
Accepts values JSON written by Electron (via workflow:start).
Runs: GroundingDINO step-1 → [OpenCLIP semantic step-2] → [Spatial filter step-3]
Prints RESULT_JSON:<json> at the end.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root so all sub-packages are importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent        # app/workflows/sam_gdino_wizard_detect -> app/
_REPO_ROOT = _APP_DIR.parent   # app/ -> repo_root
for _p in [str(_REPO_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Entry point (called by Electron via: python -m workflows.run <id> --values-json <file>)
# ---------------------------------------------------------------------------

def main(values: dict) -> None:
    image_paths: list[str] = values.get("image_paths") or []
    recursive: bool = bool(values.get("recursive", False))
    box_threshold: float = float(values.get("box_threshold", 0.16))
    text_threshold: float = float(values.get("text_threshold", 0.16))
    max_dets: int = int(values.get("max_dets", 80))
    tiled_threshold: int = int(values.get("tiled_threshold", 512))
    tile_scales_raw: str = str(values.get("tile_scales", "small,medium,large"))
    tile_scales: list[str] = [s.strip() for s in tile_scales_raw.split(",") if s.strip()]
    prompt_groups_raw: str = str(values.get("prompt_groups", "[]"))
    gdino_checkpoint: str = str(values.get("gdino_checkpoint", "") or "")
    device: str = str(values.get("device", "auto"))
    semantic_enabled: bool = bool(values.get("semantic_enabled", True))
    semantic_model: str = str(values.get("semantic_model", "ViT-B-32"))
    semantic_pretrained: str = str(values.get("semantic_pretrained", "laion2b_s34b_b79k"))
    semantic_batch_size: int = int(values.get("semantic_batch_size", 16))
    spatial_enabled: bool = bool(values.get("spatial_enabled", True))
    spatial_iou: float = float(values.get("spatial_iou", 0.5))
    spatial_containment: float = float(values.get("spatial_containment", 0.8))

    try:
        prompt_groups = json.loads(prompt_groups_raw)
    except Exception:
        prompt_groups = [{"name": "crack", "prompt": "crack"}]

    session_id = str(uuid.uuid4())[:8]
    runs_root = _REPO_ROOT / ".tmp" / "sam_gdino_wizard" / session_id
    runs_root.mkdir(parents=True, exist_ok=True)

    step1_db = runs_root / "step1.sqlite3"
    print(f"[wizard-detect] session={session_id}", flush=True)
    print(f"[wizard-detect] images={len(image_paths)}, recursive={recursive}", flush=True)

    # ── Collect image files ──────────────────────────────────────────────────
    from object_detection.damage_scan.pipeline import IMAGE_EXTS
    all_images: list[Path] = []
    for raw in image_paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            all_images.append(p)
        elif p.is_dir():
            gen = p.rglob("*") if recursive else p.glob("*")
            all_images.extend(q for q in sorted(gen) if q.is_file() and q.suffix.lower() in IMAGE_EXTS)

    if not all_images:
        print("RESULT_JSON:" + json.dumps({"boxes_by_image": {}, "suspect_by_image": {}, "db_path": None}), flush=True)
        return

    print(f"[wizard-detect] total images={len(all_images)}", flush=True)

    # Build temporary input_dir containing symlinks (or just pass first image dir)
    # We'll use absolute path mode so each image maps to its absolute path.
    # Damage scan needs an input_dir, so use the parent of the first image as a
    # starting point, but then override using absolute mode.
    # For simplicity, write a manifest and process images one-by-one via service.

    # ── Step 1: GroundingDINO scan ───────────────────────────────────────────
    from object_detection.damage_scan.pipeline import DamageScanConfig, DamageScanPipeline
    from object_detection.damage_scan.prompts import PROMPT_SPECS, PromptSpec

    # Override prompt specs with wizard groups
    custom_specs: dict[str, PromptSpec] = {}
    for g in prompt_groups:
        key = str(g.get("name", "custom")).lower().replace(" ", "_")
        raw_prompt = str(g.get("prompt", key))
        custom_specs[key] = PromptSpec(
            key=key,
            prompt=raw_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            tile_bias="medium",
            max_side_ratio=0.5,
            max_area_ratio=0.5,
            min_component_area=20,
            nms_iou=0.45,
            fuse_iou=0.5,
        )

    # Temporarily monkey-patch PROMPT_SPECS / PROMPT_ORDER. The pipeline module
    # imports these by name so we patch on _pipeline_mod, not _prompts_mod.
    import object_detection.damage_scan.pipeline as _pipeline_mod
    _orig_pipe_specs = getattr(_pipeline_mod, "PROMPT_SPECS", None)
    _orig_pipe_order = getattr(_pipeline_mod, "PROMPT_ORDER", None)
    _pipeline_mod.PROMPT_SPECS = custom_specs
    _pipeline_mod.PROMPT_ORDER = tuple(custom_specs.keys())

    try:
        # DamageScanPipeline expects a single input_dir; use first image's parent
        # but set store_image_path_mode=absolute so paths are unique
        cfg = DamageScanConfig(
            input_dir=all_images[0].parent,
            db_path=step1_db,
            checkpoint=gdino_checkpoint,
            device=device,
            recursive=False,
            limit=0,
            save_overlays=False,
            store_image_path_mode="absolute",
        )
        pipeline = DamageScanPipeline(cfg)
        # Inject images directly by patching iter_images locally
        import object_detection.damage_scan.pipeline as _pm
        _orig_iter = _pm.iter_images

        def _patched_iter(input_dir, *, recursive, limit=0):
            return list(all_images)

        _pm.iter_images = _patched_iter
        try:
            run_id = pipeline.run(log_fn=lambda m: print(f"[step1] {m}", flush=True))
        finally:
            _pm.iter_images = _orig_iter
            pipeline.close()
    finally:
        if _orig_pipe_specs is not None:
            _pipeline_mod.PROMPT_SPECS = _orig_pipe_specs
        if _orig_pipe_order is not None:
            _pipeline_mod.PROMPT_ORDER = _orig_pipe_order

    print(f"[wizard-detect] step1 run_id={run_id}", flush=True)

    # ── Step 2: Semantic relabel ─────────────────────────────────────────────
    semantic_run_id = None
    if semantic_enabled:
        step2_db = runs_root / "step2.sqlite3"
        shutil.copy2(step1_db, step2_db)
        try:
            sys.path.insert(0, str(_REPO_ROOT / "semi-labeling" / "step2_sematic"))
            from pipeline import Step2SemanticConfig, Step2SemanticPipeline  # type: ignore
            s2cfg = Step2SemanticConfig(
                db_path=step2_db,
                source_run_id=run_id,
                model_name=semantic_model,
                pretrained=semantic_pretrained,
                device=device,
                batch_size=semantic_batch_size,
            )
            s2 = Step2SemanticPipeline(s2cfg)
            semantic_run_id = s2.run(log_fn=lambda m: print(f"[step2] {m}", flush=True))
            print(f"[wizard-detect] step2 semantic_run_id={semantic_run_id}", flush=True)
            step1_db = step2_db  # downstream uses step2 output
        except Exception as exc:
            print(f"[wizard-detect] WARNING: semantic step failed: {exc}", flush=True)

    # ── Step 3: Spatial filter ───────────────────────────────────────────────
    filtered_db = step1_db
    suspect_db = None
    if spatial_enabled:
        step3_dir = runs_root / "step3"
        step3_dir.mkdir(parents=True, exist_ok=True)
        try:
            filter_script = _REPO_ROOT / "semi-labeling" / "step3_spatial_filter" / "filter_duplicates.py"
            import subprocess, sys as _sys
            result = subprocess.run(
                [_sys.executable, str(filter_script),
                 "--source-db", str(step1_db),
                 "--output-dir", str(step3_dir),
                 "--semantic-run-id", str(semantic_run_id or "latest"),
                 "--iou-threshold", str(spatial_iou),
                 "--containment-threshold", str(spatial_containment),
                 ],
                capture_output=True, text=True
            )
            print(result.stdout, flush=True)
            if result.returncode == 0:
                f = step3_dir / "filtered.sqlite3"
                s = step3_dir / "suspect.sqlite3"
                if f.exists():
                    filtered_db = f
                if s.exists():
                    suspect_db = s
                print(f"[wizard-detect] step3 filtered={filtered_db}", flush=True)
            else:
                print(f"[wizard-detect] WARNING: spatial filter failed: {result.stderr[:300]}", flush=True)
        except Exception as exc:
            print(f"[wizard-detect] WARNING: spatial filter error: {exc}", flush=True)

    # ── Read boxes from final DB ─────────────────────────────────────────────
    def _read_boxes(db_path: Path, semantic_run_id_val) -> dict:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        by_image: dict[str, list] = {}
        try:
            # Base query: join detections + images
            rows = conn.execute("""
                SELECT d.detection_id, d.label, d.score, d.x1, d.y1, d.x2, d.y2,
                       i.path AS image_path
                FROM detections d JOIN images i ON i.image_id = d.image_id
                WHERE d.stage = 'final'
                ORDER BY i.path, d.score DESC
            """).fetchall()
            for row in rows:
                img = str(row["image_path"])
                if img not in by_image:
                    by_image[img] = []
                box: dict = {
                    "detection_id": row["detection_id"],
                    "label": row["label"], "score": row["score"],
                    "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
                }
                # Try to attach semantic info if available
                if semantic_run_id_val:
                    try:
                        sem = conn.execute("""
                            SELECT predicted_label, predicted_probability_pct
                            FROM openclip_semantic_results
                            WHERE source_detection_id=? AND semantic_run_id=?
                            ORDER BY predicted_probability_pct DESC LIMIT 1
                        """, (row["detection_id"], semantic_run_id_val)).fetchone()
                        if sem:
                            box["semantic_label"] = sem["predicted_label"]
                            box["semantic_prob"] = sem["predicted_probability_pct"] / 100.0
                    except Exception:
                        pass
                by_image[img].append(box)
        finally:
            conn.close()
        return by_image

    boxes_by_image = _read_boxes(filtered_db, semantic_run_id)
    suspect_by_image: dict = {}
    if suspect_db and suspect_db.exists():
        suspect_by_image = _read_boxes(suspect_db, semantic_run_id)

    result = {
        "db_path": str(filtered_db),
        "run_id": run_id,
        "semantic_run_id": semantic_run_id,
        "boxes_by_image": boxes_by_image,
        "suspect_by_image": suspect_by_image,
    }
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # When invoked via workflows/__main__.py run sam_gdino_wizard_detect --values-json <file>
    import argparse, json as _json
    p = argparse.ArgumentParser()
    p.add_argument("--values-json", required=True)
    args, _ = p.parse_known_args()
    with open(args.values_json, encoding="utf-8") as f:
        main(_json.load(f))
