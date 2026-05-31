from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from shared.db.source_store import SourceDetection


@dataclass(frozen=True)
class CropViewSpec:
    name: str
    padding_ratio: float


@dataclass(frozen=True)
class CropView:
    result_id: int
    view_name: str
    crop_path: str
    image_rel_path: str
    x1: float
    y1: float
    x2: float
    y2: float
    source: str
    crop_hash: str
    crop_width: int
    crop_height: int
    padding_ratio: float
    status: str = "ok"
    error_message: str | None = None


DEFAULT_VIEW_SPECS: tuple[CropViewSpec, ...] = (
    CropViewSpec("tight", 0.0),
    CropViewSpec("pad10", 0.10),
    CropViewSpec("pad25", 0.25),
    CropViewSpec("context", 0.75),
)


def parse_view_specs(raw: str) -> tuple[CropViewSpec, ...]:
    if not str(raw or "").strip():
        return DEFAULT_VIEW_SPECS
    by_name = {item.name: item for item in DEFAULT_VIEW_SPECS}
    specs: list[CropViewSpec] = []
    for name in [item.strip() for item in str(raw).split(",") if item.strip()]:
        if name not in by_name:
            raise ValueError(f"Unknown crop view: {name}. Expected one of: {', '.join(by_name)}")
        specs.append(by_name[name])
    return tuple(specs)


def generate_crop_views(
    detections: list[SourceDetection],
    *,
    image_root: Path | None,
    crop_dir: Path,
    view_specs: tuple[CropViewSpec, ...] = DEFAULT_VIEW_SPECS,
) -> tuple[list[CropView], dict[int, str]]:
    crop_dir = Path(crop_dir).expanduser().resolve()
    crop_dir.mkdir(parents=True, exist_ok=True)
    views: list[CropView] = []
    errors: dict[int, str] = {}
    for detection in detections:
        try:
            views.extend(_generate_detection_views(detection, image_root=image_root, crop_dir=crop_dir, view_specs=view_specs))
        except (FileNotFoundError, UnidentifiedImageError, ValueError, OSError) as exc:
            errors[detection.result_id] = str(exc)
    return views, errors


def _generate_detection_views(
    detection: SourceDetection,
    *,
    image_root: Path | None,
    crop_dir: Path,
    view_specs: tuple[CropViewSpec, ...],
) -> list[CropView]:
    image_path = resolve_image_path(detection, image_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        output: list[CropView] = []
        for spec in view_specs:
            box = padded_box(detection, width=width, height=height, padding_ratio=spec.padding_ratio)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop = rgb.crop((x1, y1, x2, y2))
            crop_hash, encoded = encode_png_with_hash(crop)
            view_dir = crop_dir / spec.name
            view_dir.mkdir(parents=True, exist_ok=True)
            path = view_dir / f"{detection.result_id}_{spec.name}_{crop_hash[:12]}.png"
            if not path.is_file():
                path.write_bytes(encoded)
            output.append(
                CropView(
                    result_id=detection.result_id,
                    view_name=spec.name,
                    crop_path=str(path),
                    image_rel_path=detection.image_rel_path,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    source="resemi_multicrop",
                    crop_hash=crop_hash,
                    crop_width=int(crop.width),
                    crop_height=int(crop.height),
                    padding_ratio=float(spec.padding_ratio),
                )
            )
        return output


def resolve_image_path(detection: SourceDetection, image_root: Path | None) -> Path:
    candidates: list[Path] = []
    rel_path = str(detection.image_rel_path or "").strip()
    stored_path = str(detection.image_path or "").strip()
    source_input_dir = Path(str(detection.source_input_dir or "")).expanduser()

    if image_root is not None:
        root = Path(image_root).expanduser().resolve()
        if rel_path:
            candidates.append(root / rel_path)
        if stored_path:
            candidates.append(root / Path(stored_path).name)
    if stored_path:
        stored = Path(stored_path).expanduser()
        candidates.append(stored if stored.is_absolute() else source_input_dir / stored_path)
    if rel_path:
        candidates.append(source_input_dir / rel_path)
    if stored_path:
        candidates.append(source_input_dir / Path(stored_path).name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if image_root is not None and rel_path:
        return (Path(image_root).expanduser().resolve() / rel_path).resolve()
    return (source_input_dir / rel_path).expanduser().resolve()


def padded_box(detection: SourceDetection, *, width: int, height: int, padding_ratio: float) -> tuple[int, int, int, int] | None:
    box_width = max(0.0, float(detection.x2) - float(detection.x1))
    box_height = max(0.0, float(detection.y2) - float(detection.y1))
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    pad_x = box_width * float(padding_ratio)
    pad_y = box_height * float(padding_ratio)
    x1 = max(0, int(float(detection.x1) - pad_x))
    y1 = max(0, int(float(detection.y1) - pad_y))
    x2 = min(int(width), int(float(detection.x2) + pad_x + 0.999999))
    y2 = min(int(height), int(float(detection.y2) + pad_y + 0.999999))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def encode_png_with_hash(image: Image.Image) -> tuple[str, bytes]:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return hashlib.sha256(data).hexdigest(), data
