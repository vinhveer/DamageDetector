from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .crops import clamp_xyxy
from .reports import write_prediction_report


def _normalize_prediction_box(item: dict[str, Any]) -> tuple[list[float] | None, str, float]:
    raw_box = item.get("box") or item.get("bbox") or item.get("xyxy")
    label = str(item.get("label") or item.get("class") or item.get("name") or "")
    score = float(item.get("score") or item.get("confidence") or item.get("conf") or 0.0)
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return None, label, score
    return [float(v) for v in raw_box[:4]], label, score


def _canonical_label(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").strip().lower().split())


def _field_label(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned or "unknown"


def _safe_preview_name(row_idx: int, image_path: str, suffix: str = ".png") -> str:
    stem = Path(image_path).stem or "image"
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)[:80]
    return f"{row_idx + 1:06d}_{stem}{suffix}"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_or_build_prototypes(
    *,
    runner: Any,
    checkpoint_path: str,
    prototype_dir: str | Path,
    device: str,
    batch_size: int,
    cache_path: str | Path | None,
) -> dict[str, Any]:
    from torch_runtime import get_torch, select_device_str

    torch = get_torch()
    actual_device = select_device_str(device)
    root = Path(prototype_dir).expanduser().resolve()
    cache = Path(cache_path).expanduser().resolve() if cache_path else root / "prototype_bank.pt"
    checkpoint = str(checkpoint_path or "").strip()
    if cache.is_file():
        try:
            payload = torch.load(cache, map_location=actual_device)
            if isinstance(payload, dict) and payload.get("checkpoint_path") == checkpoint and payload.get("prototype_dir") == str(root):
                payload["matrix"] = payload["matrix"].to(actual_device)
                payload["cache_path"] = str(cache)
                payload["cache_hit"] = True
                return payload
        except Exception:
            pass

    built = runner._load_prototypes(  # noqa: SLF001 - shared internal runner helper avoids duplicate DINOv2 code.
        checkpoint_path=checkpoint,
        prototype_dir=str(root),
        include_labels=None,
        device_preference=device,
        batch_size=int(batch_size),
    )
    payload = {
        "checkpoint_path": checkpoint,
        "prototype_dir": str(root),
        "labels": list(built["labels"]),
        "counts": list(built["counts"]),
        "matrix": built["matrix"].detach().cpu(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    payload["matrix"] = payload["matrix"].to(actual_device)
    payload["cache_path"] = str(cache)
    payload["cache_hit"] = False
    return payload


def _decision_for_row(
    *,
    detector_label: str,
    nearest_label: str,
    best_similarity: float,
    predicted_class_similarity: float,
    class_margin: float,
    mode: str,
    threshold: float,
    reject_threshold: float,
    margin_threshold: float,
) -> str:
    if best_similarity < reject_threshold:
        return "reject"
    normalized_mode = str(mode or "coverage").strip().lower()
    if normalized_mode in {"class-consistency", "class_consistency"}:
        if (
            _canonical_label(detector_label) == _canonical_label(nearest_label)
            and predicted_class_similarity >= threshold
            and class_margin >= margin_threshold
        ):
            return "accept"
        return "review"
    if best_similarity >= threshold:
        return "accept"
    return "review"


def semantic_validate_predictions(
    *,
    predictions: list[dict[str, Any]],
    output_dir: str | Path,
    prototype_dir: str | Path | None = None,
    dinov2_checkpoint: str | None = None,
    device: str = "auto",
    batch_size: int = 16,
    threshold: float = 0.75,
    reject_threshold: float = 0.50,
    margin_threshold: float = 0.05,
    decision_mode: str = "coverage",
    expand_ratio: float = 0.05,
    prototype_cache_path: str | Path | None = None,
    expected_image_paths: list[str | Path] | None = None,
    expected_per_class: dict[str, list[str | Path]] | None = None,
    save_previews: bool = True,
    preview_limit: int = 200,
) -> dict[str, Any]:
    from PIL import Image

    rows: list[dict[str, Any]] = []
    crops = []
    crop_row_indices: list[int] = []
    skipped = Counter()

    for pred in predictions:
        image_path = Path(str(pred.get("image_path") or pred.get("path") or "")).expanduser()
        if not image_path.is_file():
            skipped["missing_image"] += 1
            continue
        box, label, score = _normalize_prediction_box(pred)
        if box is None:
            skipped["missing_box"] += 1
            continue
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            clamped = clamp_xyxy(box, width, height, expand_ratio=float(expand_ratio))
            if clamped is None:
                skipped["invalid_box"] += 1
                continue
            crop = rgb.crop(clamped)
        row = {
            "image_path": str(image_path.resolve()),
            "image_width": width,
            "image_height": height,
            "bbox_xyxy": " ".join(str(v) for v in clamped),
            "detector_label": label,
            "detector_score": score,
            "nearest_label": "",
            "best_similarity": 0.0,
            "decision": "unvalidated",
        }
        rows.append(row)
        crops.append(crop)
        crop_row_indices.append(len(rows) - 1)

    prototype_cache: dict[str, Any] | None = None

    if crops and prototype_dir and dinov2_checkpoint:
        from object_detection.dinov2.dinov2_prototypes import DinoV2PrototypeRunner
        from torch_runtime import get_torch

        torch = get_torch()
        runner = DinoV2PrototypeRunner()
        prototype_cache = _load_or_build_prototypes(
            runner=runner,
            checkpoint_path=str(dinov2_checkpoint),
            prototype_dir=str(prototype_dir),
            device=str(device),
            batch_size=int(batch_size),
            cache_path=prototype_cache_path,
        )
        crop_embeddings = runner._embed_images(  # noqa: SLF001 - shared internal runner helper avoids duplicate DINOv2 code.
            checkpoint_path=str(dinov2_checkpoint),
            images=crops,
            device_preference=str(device),
            batch_size=int(batch_size),
        )
        matrix = prototype_cache["matrix"]
        similarities = crop_embeddings @ matrix.T
        labels = list(prototype_cache["labels"])
        counts = list(prototype_cache["counts"])
        top_k = max(1, min(3, len(labels)))
        top_scores, top_ids = torch.topk(similarities, k=top_k, dim=-1)
        for local_idx, row_idx in enumerate(crop_row_indices):
            row_scores = top_scores[local_idx].detach().cpu().tolist()
            row_ids = top_ids[local_idx].detach().cpu().tolist()
            top_predictions = []
            sim_values = similarities[local_idx].detach().cpu().tolist()
            sim_by_label: dict[str, float] = {}
            for label, sim in zip(labels, sim_values):
                sim_by_label[str(label)] = float(sim)
                rows[row_idx][f"sim_{_field_label(label)}"] = float(sim)
            for score, proto_idx in zip(row_scores, row_ids):
                top_predictions.append({
                    "label": str(labels[int(proto_idx)]),
                    "similarity": float(score),
                    "support_count": int(counts[int(proto_idx)]),
                    "prototype_id": int(proto_idx),
                })
            nearest = top_predictions[0]["label"] if top_predictions else ""
            best_sim = float(top_predictions[0]["similarity"] if top_predictions else 0.0)
            detector_label = str(rows[row_idx].get("detector_label") or "")
            detector_key = _canonical_label(detector_label)
            predicted_class_similarity = 0.0
            other_sims = []
            for label, sim in sim_by_label.items():
                if _canonical_label(label) == detector_key:
                    predicted_class_similarity = float(sim)
                else:
                    other_sims.append(float(sim))
            max_other = max(other_sims) if other_sims else 0.0
            class_margin = predicted_class_similarity - max_other
            rows[row_idx]["nearest_label"] = nearest
            rows[row_idx]["best_similarity"] = best_sim
            rows[row_idx]["predicted_class_similarity"] = predicted_class_similarity
            rows[row_idx]["class_margin"] = class_margin
            rows[row_idx]["class_agreement"] = _canonical_label(detector_label) == _canonical_label(nearest)
            rows[row_idx]["decision"] = _decision_for_row(
                detector_label=detector_label,
                nearest_label=nearest,
                best_similarity=best_sim,
                predicted_class_similarity=predicted_class_similarity,
                class_margin=class_margin,
                mode=decision_mode,
                threshold=float(threshold),
                reject_threshold=float(reject_threshold),
                margin_threshold=float(margin_threshold),
            )
            rows[row_idx]["top_predictions"] = json.dumps(top_predictions, ensure_ascii=False)
    elif crops:
        for row_idx in crop_row_indices:
            rows[row_idx]["decision"] = "detector_only"

    output_root = Path(output_dir).expanduser().resolve()
    preview_counts = Counter()
    if save_previews:
        preview_root = output_root / "preview"
        for dirname in ("accepted", "low_similarity", "class_mismatch", "review", "no_detection_images"):
            (preview_root / dirname).mkdir(parents=True, exist_ok=True)
        for crop, row_idx in zip(crops, crop_row_indices):
            if sum(preview_counts.values()) >= int(preview_limit):
                break
            row = rows[row_idx]
            decision = str(row.get("decision") or "")
            detector_label = str(row.get("detector_label") or "")
            nearest_label = str(row.get("nearest_label") or "")
            if decision == "accept":
                dirname = "accepted"
            elif decision == "reject":
                dirname = "low_similarity"
            elif nearest_label and _canonical_label(detector_label) != _canonical_label(nearest_label):
                dirname = "class_mismatch"
            else:
                dirname = "review"
            crop.save(preview_root / dirname / _safe_preview_name(row_idx, str(row.get("image_path") or "")))
            preview_counts[dirname] += 1

    expected = {str(Path(path).expanduser().resolve()) for path in (expected_image_paths or [])}
    predicted_images = {str(row.get("image_path") or "") for row in rows}
    if not expected:
        expected = set(predicted_images)
    accepted_images = {str(row.get("image_path") or "") for row in rows if str(row.get("decision") or "") == "accept"}
    no_detection_images = sorted(expected - predicted_images)
    if save_previews and no_detection_images:
        preview_dir = output_root / "preview" / "no_detection_images"
        for idx, image_path in enumerate(no_detection_images[: max(0, int(preview_limit) - sum(preview_counts.values()))], start=1):
            src = Path(image_path)
            if src.is_file():
                _link_or_copy(src, preview_dir / f"{idx:06d}_{src.name}")
                preview_counts["no_detection_images"] += 1

    decisions = Counter(str(row.get("decision", "")) for row in rows)
    label_counts = Counter(str(row.get("nearest_label") or row.get("detector_label") or "") for row in rows)
    total_predictions = len(rows)
    accepted = int(decisions.get("accept", 0))
    reviewed = int(decisions.get("review", 0))
    rejected = int(decisions.get("reject", 0))
    class_agreement_rows = [row for row in rows if row.get("nearest_label")]
    class_agreement_count = sum(1 for row in class_agreement_rows if bool(row.get("class_agreement")))
    image_damage_area: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.get("decision") != "accept":
            continue
        try:
            image_path = str(row["image_path"])
            x1, y1, x2, y2 = [float(value) for value in str(row["bbox_xyxy"]).split()]
            width = int(row.get("image_width") or 0)
            height = int(row.get("image_height") or 0)
            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if width * height > 0:
                image_damage_area[image_path] += bbox_area / (width * height)
        except Exception:
            pass
    summary = {
        "total_predictions": total_predictions,
        "skipped": dict(skipped),
        "decisions": dict(decisions),
        "labels": dict(label_counts),
        "threshold": float(threshold),
        "reject_threshold": float(reject_threshold),
        "margin_threshold": float(margin_threshold),
        "decision_mode": str(decision_mode),
        "expand_ratio": float(expand_ratio),
        "prototype_dir": str(prototype_dir) if prototype_dir else None,
        "prototype_cache_path": str(prototype_cache.get("cache_path")) if prototype_cache else None,
        "prototype_cache_hit": bool(prototype_cache.get("cache_hit")) if prototype_cache else None,
        "dinov2_checkpoint": str(dinov2_checkpoint) if dinov2_checkpoint else None,
        "metrics": {
            "prediction_semantic_pass_rate": accepted / total_predictions if total_predictions else 0.0,
            "class_agreement_rate": class_agreement_count / len(class_agreement_rows) if class_agreement_rows else 0.0,
            "low_similarity_rate": rejected / total_predictions if total_predictions else 0.0,
            "review_queue_size": reviewed + rejected + int(decisions.get("detector_only", 0)),
            "image_level_coverage": len(accepted_images) / len(expected) if expected else 0.0,
            "images_expected": len(expected),
            "images_with_prediction": len(predicted_images),
            "images_with_accepted_prediction": len(accepted_images),
            "miss_candidates": len(no_detection_images),
            "damage_area": {
                "mean_ratio": sum(image_damage_area.values()) / len(image_damage_area) if image_damage_area else 0.0,
                "per_image": dict(image_damage_area),
            },
        },
        "no_detection_images": no_detection_images,
        "preview_counts": dict(preview_counts),
    }
    if expected_per_class:
        coverage_by_class = {}
        for label, expected_paths in expected_per_class.items():
            expected_set = {str(Path(path).expanduser().resolve()) for path in expected_paths}
            accepted_for_class = {
                str(row.get("image_path") or "")
                for row in rows
                if row.get("decision") == "accept"
                and _canonical_label(str(row.get("nearest_label") or row.get("detector_label") or "")) == _canonical_label(label)
            }
            coverage_by_class[label] = len(accepted_for_class & expected_set) / len(expected_set) if expected_set else 0.0
        summary["metrics"]["coverage_by_class"] = coverage_by_class
    paths = write_prediction_report(output_dir=output_dir, rows=rows, summary=summary)
    return {**summary, "outputs": paths}
