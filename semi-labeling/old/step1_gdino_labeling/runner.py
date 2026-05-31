from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from object_detection.dino.client import get_dino_service
from object_detection.dino.engine import default_gdino_checkpoint
from outputs import rel_image_key, write_manifest, write_readme, write_summary_csv
from prompts import PromptGroup, combined_queries, match_prompt_groups
from store import (
    delete_existing_detections,
    ensure_schema,
    insert_detections,
    insert_image_metadata,
    insert_run_metadata,
    record_image_run,
)
from tools.gdino_detect_folder import (
    _IMAGE_EXTS,
    _iter_images,
    _max_dim_fast,
    _rows_from_detections,
)


def collect_images(input_dir: Path, *, recursive_find: bool, limit: int) -> list[Path]:
    images = _iter_images(input_dir, recursive=recursive_find)
    if int(limit or 0) > 0:
        images = images[: int(limit)]
    return images


def resolve_checkpoint(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint or "").strip() or str(default_gdino_checkpoint() or "").strip()
    if not checkpoint:
        raise RuntimeError("No GroundingDINO checkpoint available. Pass --checkpoint or download the default model.")
    return checkpoint


def build_params(args, *, checkpoint: str, merged_queries: list[str]) -> dict:
    return {
        "gdino_checkpoint": checkpoint,
        "gdino_config_id": "auto",
        "text_queries": merged_queries,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "max_dets": int(args.max_dets),
        "device": args.device,
        "recursive_tile_scales": [s.strip() for s in str(args.tile_scales).split(",") if s.strip()],
    }


def run_semi_label_job(
    *,
    args,
    input_dir: Path,
    output_dir: Path,
    prompt_groups: list[PromptGroup],
    request_names: list[str],
) -> int:
    sqlite_path = output_dir / "boxes.sqlite3"
    readme_path = output_dir / "README.txt"
    prompt_groups_path = output_dir / "prompt_groups.json"
    summary_csv_path = output_dir / "image_runs.csv"

    images = collect_images(input_dir, recursive_find=bool(args.recursive_find), limit=int(args.limit or 0))
    if not images:
        print(f"No images found in {input_dir} with extensions: {sorted(_IMAGE_EXTS)}", flush=True)
        return 2

    checkpoint = resolve_checkpoint(args.checkpoint)
    merged_queries = combined_queries(prompt_groups)
    params = build_params(args, checkpoint=checkpoint, merged_queries=merged_queries)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_rows: list[tuple] = []

    conn = sqlite3.connect(str(sqlite_path))
    service = get_dino_service()
    try:
        ensure_schema(conn)
        insert_run_metadata(
            conn,
            run_id=run_id,
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint=checkpoint,
            device=args.device,
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            max_dets=int(args.max_dets),
            tiled_threshold=int(args.tiled_threshold),
            recursive_find=bool(args.recursive_find),
            recursive_max_depth=int(args.recursive_max_depth),
            recursive_min_box_px=int(args.recursive_min_box_px),
            image_count=len(images),
            request_names=request_names,
            prompt_groups=prompt_groups,
        )
        write_manifest(
            manifest_path=prompt_groups_path,
            run_id=run_id,
            prompt_groups=prompt_groups,
            request_names=request_names,
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint=checkpoint,
            total_images=len(images),
        )
        write_readme(readme_path)

        for group in prompt_groups:
            print(
                f"Prompt group {group.group_id}/{len(prompt_groups)}: {group.name} -> {group.prompt_text}",
                flush=True,
            )

        total_jobs = len(images)
        for job_idx, img_path in enumerate(images, start=1):
            _process_image(
                args=args,
                conn=conn,
                service=service,
                prompt_groups=prompt_groups,
                request_names=request_names,
                merged_queries=merged_queries,
                params=params,
                input_dir=input_dir,
                output_dir=output_dir,
                img_path=img_path,
                job_idx=job_idx,
                total_jobs=total_jobs,
                run_id=run_id,
                summary_rows=summary_rows,
            )

        write_summary_csv(summary_csv_path, summary_rows)
        print(f"SQLite saved to: {sqlite_path}", flush=True)
        print(f"Pass output saved to: {output_dir}", flush=True)
        print(f"Summary CSV: {summary_csv_path}", flush=True)
        return 0
    finally:
        service.close()
        conn.close()


def _process_image(
    *,
    args,
    conn: sqlite3.Connection,
    service,
    prompt_groups: list[PromptGroup],
    request_names: list[str],
    merged_queries: list[str],
    params: dict,
    input_dir: Path,
    output_dir: Path,
    img_path: Path,
    job_idx: int,
    total_jobs: int,
    run_id: str,
    summary_rows: list[tuple],
) -> None:
    from PIL import Image

    with Image.open(img_path) as pil_img:
        width, height = pil_img.size

    image_rel_path = rel_image_key(input_dir, img_path)
    insert_image_metadata(
        conn,
        run_id=run_id,
        image_rel_path=image_rel_path,
        image_path=img_path,
        width=int(width),
        height=int(height),
    )
    conn.commit()

    max_dim = _max_dim_fast(img_path)
    use_tiled = int(max_dim) > int(args.tiled_threshold)
    mode = "tiled" if use_tiled else "single"
    print(f"[{job_idx}/{total_jobs}] {img_path.name} mode={mode}", flush=True)
    log_fn = (lambda line: print(line, flush=True)) if bool(args.verbose_logs) else None

    try:
        result = _predict_image(
            service=service,
            img_path=img_path,
            params=params,
            merged_queries=merged_queries,
            use_tiled=use_tiled,
            recursive_max_depth=int(args.recursive_max_depth),
            recursive_min_box_px=int(args.recursive_min_box_px),
            log_fn=log_fn,
        )
        _save_success_outputs(
            args=args,
            conn=conn,
            prompt_groups=prompt_groups,
            request_names=request_names,
            input_dir=input_dir,
            output_dir=output_dir,
            img_path=img_path,
            image_rel_path=image_rel_path,
            width=int(width),
            height=int(height),
            run_id=run_id,
            mode=mode,
            detections=list(result.get("detections") or []),
            summary_rows=summary_rows,
        )
        conn.commit()
    except Exception as exc:
        _save_error_outputs(
            conn=conn,
            prompt_groups=prompt_groups,
            input_dir=input_dir,
            output_dir=output_dir,
            img_path=img_path,
            image_rel_path=image_rel_path,
            run_id=run_id,
            mode=mode,
            exc=exc,
            summary_rows=summary_rows,
        )
        conn.commit()
        print(f"  error: {img_path.name}: {exc}", flush=True)


def _predict_image(
    *,
    service,
    img_path: Path,
    params: dict,
    merged_queries: list[str],
    use_tiled: bool,
    recursive_max_depth: int,
    recursive_min_box_px: int,
    log_fn,
):
    if use_tiled:
        return service.call(
            "recursive_detect",
            {
                "image_path": str(img_path),
                "params": params,
                "target_labels": merged_queries,
                "max_depth": int(recursive_max_depth),
                "min_box_px": int(recursive_min_box_px),
            },
            log_fn=log_fn,
        )
    return service.call(
        "predict",
        {"image_path": str(img_path), "params": params},
        log_fn=log_fn,
    )


def _group_detections_by_prompt(
    detections: list[dict],
    prompt_groups: list[PromptGroup],
) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {int(group.group_id): [] for group in prompt_groups}
    for det in detections:
        group_ids = match_prompt_groups(det.get("label") or "", prompt_groups)
        for group_id in group_ids:
            grouped.setdefault(int(group_id), []).append(dict(det))
    return grouped


def _save_success_outputs(
    *,
    args,
    conn: sqlite3.Connection,
    prompt_groups: list[PromptGroup],
    request_names: list[str],
    input_dir: Path,
    output_dir: Path,
    img_path: Path,
    image_rel_path: str,
    width: int,
    height: int,
    run_id: str,
    mode: str,
    detections: list[dict],
    summary_rows: list[tuple],
) -> None:
    grouped_detections = _group_detections_by_prompt(detections, prompt_groups)

    for group in prompt_groups:
        current_artifact_dir = output_dir
        group_detections = grouped_detections.get(int(group.group_id), [])
        rows = _rows_from_detections(
            image_path=img_path,
            detections=group_detections,
            width=int(width),
            height=int(height),
        )

        delete_existing_detections(
            conn,
            run_id=run_id,
            prompt_group_id=group.group_id,
            image_rel_path=image_rel_path,
        )
        insert_detections(
            conn,
            run_id=run_id,
            request_names=request_names,
            prompt_group_id=group.group_id,
            image_rel_path=image_rel_path,
            image_path=img_path,
            rows=rows,
            artifact_dir=current_artifact_dir,
        )
        record_image_run(
            conn,
            run_id=run_id,
            prompt_group_id=group.group_id,
            image_rel_path=image_rel_path,
            mode=mode,
            artifact_dir=current_artifact_dir,
            status="ok",
            detection_count=len(rows),
        )
        summary_rows.append(
            (
                group.group_id,
                group.name,
                image_rel_path,
                "ok",
                mode,
                len(rows),
                str(current_artifact_dir),
                "",
                "",
            )
        )


def _save_error_outputs(
    *,
    conn: sqlite3.Connection,
    prompt_groups: list[PromptGroup],
    input_dir: Path,
    output_dir: Path,
    img_path: Path,
    image_rel_path: str,
    run_id: str,
    mode: str,
    exc: Exception,
    summary_rows: list[tuple],
) -> None:
    for group in prompt_groups:
        current_artifact_dir = output_dir
        delete_existing_detections(
            conn,
            run_id=run_id,
            prompt_group_id=group.group_id,
            image_rel_path=image_rel_path,
        )
        record_image_run(
            conn,
            run_id=run_id,
            prompt_group_id=group.group_id,
            image_rel_path=image_rel_path,
            mode=mode,
            artifact_dir=current_artifact_dir,
            status="error",
            detection_count=0,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )
        summary_rows.append(
            (
                group.group_id,
                group.name,
                image_rel_path,
                "error",
                mode,
                0,
                str(current_artifact_dir),
                exc.__class__.__name__,
                str(exc),
            )
        )
