from __future__ import annotations

from dataclasses import dataclass, replace

from .models import Box, Detection, Tile


def clip_box(box: Box, *, width: int, height: int) -> Box | None:
    x1 = max(0.0, min(float(width), float(box.x1)))
    y1 = max(0.0, min(float(height), float(box.y1)))
    x2 = max(0.0, min(float(width), float(box.x2)))
    y2 = max(0.0, min(float(height), float(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(float(a.x1), float(b.x1))
    iy1 = max(float(a.y1), float(b.y1))
    ix2 = min(float(a.x2), float(b.x2))
    iy2 = min(float(a.y2), float(b.y2))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = float(a.area) + float(b.area) - inter
    return inter / union if union > 0.0 else 0.0


def expand_box(box: Box, *, ratio: float, width: int, height: int) -> Box | None:
    grow_w = float(box.width) * float(ratio)
    grow_h = float(box.height) * float(ratio)
    expanded = Box(
        x1=float(box.x1) - (grow_w / 2.0),
        y1=float(box.y1) - (grow_h / 2.0),
        x2=float(box.x2) + (grow_w / 2.0),
        y2=float(box.y2) + (grow_h / 2.0),
    )
    return clip_box(expanded, width=int(width), height=int(height))


def valid_detection(
    det: Detection,
    *,
    image_width: int,
    image_height: int,
    min_box_side: int,
    min_area_px: int,
    max_area_ratio: float,
) -> bool:
    if det.box.width < int(min_box_side) or det.box.height < int(min_box_side):
        return False
    if det.box.area < int(min_area_px):
        return False
    image_area = max(1, int(image_width) * int(image_height))
    if float(det.box.area) / float(image_area) > float(max_area_ratio):
        return False
    return True


def nms_detections(detections: list[Detection], *, iou_threshold: float, max_dets: int = 0) -> list[Detection]:
    ordered = sorted(detections, key=lambda item: float(item.score), reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        if any(box_iou(det.box, existing.box) >= float(iou_threshold) for existing in kept):
            continue
        kept.append(det)
        if int(max_dets) > 0 and len(kept) >= int(max_dets):
            break
    return kept


def fuse_detections(detections: list[Detection], *, iou_threshold: float, max_dets: int = 0) -> list[Detection]:
    remaining = sorted(detections, key=lambda item: float(item.score), reverse=True)
    fused: list[Detection] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        keep_rest: list[Detection] = []
        for other in remaining:
            if other.prompt_key == seed.prompt_key and box_iou(seed.box, other.box) >= float(iou_threshold):
                cluster.append(other)
            else:
                keep_rest.append(other)
        remaining = keep_rest
        total_weight = sum(max(1e-6, float(item.score)) for item in cluster)
        x1 = sum(float(item.box.x1) * float(item.score) for item in cluster) / total_weight
        y1 = sum(float(item.box.y1) * float(item.score) for item in cluster) / total_weight
        x2 = sum(float(item.box.x2) * float(item.score) for item in cluster) / total_weight
        y2 = sum(float(item.box.y2) * float(item.score) for item in cluster) / total_weight
        best = max(cluster, key=lambda item: float(item.score))
        fused.append(
            replace(
                best,
                box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                score=max(float(item.score) for item in cluster),
                raw={**dict(best.raw or {}), "fused_count": len(cluster)},
            )
        )
        if int(max_dets) > 0 and len(fused) >= int(max_dets):
            break
    return fused


def generate_tiles(width: int, height: int, *, tile_size: int, overlap: int) -> list[Tile]:
    tile_size = max(1, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    step = max(1, tile_size - overlap)

    def positions(limit: int) -> list[int]:
        if int(limit) <= tile_size:
            return [0]
        out: list[int] = []
        pos = 0
        last = int(limit) - tile_size
        while pos < last:
            out.append(pos)
            pos += step
        out.append(last)
        deduped: list[int] = []
        seen = set()
        for value in out:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    tiles: list[Tile] = []
    index = 1
    for y in positions(int(height)):
        for x in positions(int(width)):
            tile = Box(x1=float(x), y1=float(y), x2=float(min(width, x + tile_size)), y2=float(min(height, y + tile_size)))
            tiles.append(Tile(index=index, box=tile))
            index += 1
    return tiles


def build_adaptive_tile_plan(width: int, height: int, *, tile_bias: str) -> list[tuple[int, int]]:
    max_dim = max(int(width), int(height))
    bias = str(tile_bias or "medium").lower()
    if max_dim <= 1024:
        sizes = [256, 320, 384] if bias == "small" else [256, 384]
    elif max_dim <= 1800:
        sizes = [320, 512, 768] if bias == "small" else [384, 640, 896]
    else:
        sizes = [384, 640, 1024] if bias == "small" else [512, 768, 1152]
    plan: list[tuple[int, int]] = []
    seen = set()
    for size in sizes:
        actual_size = min(max_dim, int(size))
        if actual_size in seen:
            continue
        seen.add(actual_size)
        plan.append((actual_size, max(64, int(round(actual_size * 0.25)))))
    return plan


# --- C1: Box geometry features (pure core, self-contained) -------------------

_CONTAINS_THRESHOLD = 0.90


@dataclass(frozen=True)
class GeoInput:
    detection_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str


@dataclass(frozen=True)
class BoxGeometry:
    detection_id: int
    box_width: float
    box_height: float
    box_area: float
    area_ratio_to_image: float
    aspect_ratio: float
    elongation: float
    center_x: float
    center_y: float
    contains_count: int = 0
    contained_by_count: int = 0
    max_iou_same_label: float = 0.0
    max_containment: float = 0.0


def _width(g: GeoInput) -> float:
    return max(0.0, float(g.x2) - float(g.x1))


def _height(g: GeoInput) -> float:
    return max(0.0, float(g.y2) - float(g.y1))


def _area(g: GeoInput) -> float:
    return _width(g) * _height(g)


def elongation(width: float, height: float) -> float:
    aspect = float(width) / max(float(height), 1e-6)
    return max(aspect, 1.0 / max(aspect, 1e-6))


def intersection_area(a: GeoInput, b: GeoInput) -> float:
    iw = max(0.0, min(float(a.x2), float(b.x2)) - max(float(a.x1), float(b.x1)))
    ih = max(0.0, min(float(a.y2), float(b.y2)) - max(float(a.y1), float(b.y1)))
    return iw * ih


def iou(a: GeoInput, b: GeoInput) -> float:
    inter = intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def containment(a: GeoInput, b: GeoInput) -> float:
    inter = intersection_area(a, b)
    return inter / max(min(_area(a), _area(b)), 1e-6)


def compute_box_geometry(
    detections: list[GeoInput],
    image_width: int,
    image_height: int,
) -> list[BoxGeometry]:
    """Per-detection scalar features + O(n^2) per-image spatial pass.

    Returns one BoxGeometry per input, order-preserving.
    """
    image_area = int(image_width) * int(image_height)
    valid_image = int(image_width) > 0 and int(image_height) > 0

    contains = [0] * len(detections)
    contained_by = [0] * len(detections)
    max_iou_same = [0.0] * len(detections)
    max_cont = [0.0] * len(detections)

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            cont = containment(a, b)
            if cont > max_cont[i]:
                max_cont[i] = cont
            if cont > max_cont[j]:
                max_cont[j] = cont
            if a.label == b.label:
                ov = iou(a, b)
                if ov > max_iou_same[i]:
                    max_iou_same[i] = ov
                if ov > max_iou_same[j]:
                    max_iou_same[j] = ov
            if cont >= _CONTAINS_THRESHOLD:
                # containing detection ranks first by descending area then ascending id
                key_i = (-_area(a), int(a.detection_id))
                key_j = (-_area(b), int(b.detection_id))
                container, contained = (i, j) if key_i <= key_j else (j, i)
                contains[container] += 1
                contained_by[contained] += 1

    out: list[BoxGeometry] = []
    for idx, g in enumerate(detections):
        w, h = _width(g), _height(g)
        area = w * h
        out.append(
            BoxGeometry(
                detection_id=int(g.detection_id),
                box_width=w,
                box_height=h,
                box_area=area,
                area_ratio_to_image=(area / max(1, image_area)) if valid_image else 0.0,
                aspect_ratio=w / max(h, 1e-6),
                elongation=elongation(w, h),
                center_x=(float(g.x1) + float(g.x2)) / 2.0,
                center_y=(float(g.y1) + float(g.y2)) / 2.0,
                contains_count=contains[idx],
                contained_by_count=contained_by[idx],
                max_iou_same_label=max_iou_same[idx],
                max_containment=max_cont[idx],
            )
        )
    return out
