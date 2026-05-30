# C3 example/edge unit tests (Requirements 3.2, 3.6, 3.7, 3.8).
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import sl_imports

_pf = sl_imports.load_step("step4_class_aware_dedup", "pair_features")
dup_score_v2 = _pf.dup_score_v2
_gd = sl_imports.load_step("step4_class_aware_dedup", "greedy_dedup")
greedy_dedup = _gd.greedy_dedup
Detection = sl_imports.load_step("step4_class_aware_dedup", "source_store").Detection


def _box(x1, y1, x2, y2):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


def test_disjoint_boxes_score_zero():  # Req 3.2
    a, b = _box(0, 0, 10, 10), _box(100, 100, 110, 110)
    assert dup_score_v2(a, b, np.array([1.0, 0.0], np.float32), np.array([1.0, 0.0], np.float32)) == (0.0, 0.0, 0.0, 0.0)


def test_missing_embedding_score_zero():  # Req 3.8
    a, b = _box(0, 0, 10, 10), _box(0, 0, 10, 10)
    score, _iou, _cont, cos = dup_score_v2(a, b, np.array([1.0, 0.0], np.float32), np.zeros(2, np.float32))
    assert cos == 0.0 and score == 0.0


def test_threshold_boundary_exactly_010():  # Req 3.7
    # Fully nested (containment = 1.0) with cos = 0.10 -> score = 0.10 (>= threshold).
    small, big = _box(10, 10, 20, 20), _box(0, 0, 100, 100)
    ea = np.array([1.0, 0.0], np.float32)
    eb = np.array([0.1, float(np.sqrt(0.99))], np.float32)
    score, _iou, cont, cos = dup_score_v2(small, big, ea, eb)
    assert cont == 1.0
    assert abs(cos - 0.10) <= 1e-6
    assert abs(score - 0.10) <= 1e-6


def _det(result_id, label, box, *, detector=0.9, pct=90.0):
    x1, y1, x2, y2 = box
    return Detection(
        result_id=result_id, source_detection_id=result_id, image_id=1,
        image_rel_path="s.png", image_path="s.png", source_input_dir="",
        predicted_label=label, detector_label="damage", detector_score=detector,
        predicted_probability=pct / 100.0, predicted_probability_pct=pct,
        x1=x1, y1=y1, x2=x2, y2=y2, image_width=200, image_height=200,
    )


def _keep_map(label):
    # Long thin parent (elongation 10) + small square child (elongation 1) fully nested.
    parent = _det(1, label, (0.0, 0.0, 100.0, 10.0))
    child = _det(2, label, (10.0, 2.0, 16.0, 8.0))
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # identical -> cos 1
    decisions = greedy_dedup([parent, child], emb, dup_threshold=0.10)
    return {d.result_id: d.keep for d in decisions}


def test_elongation_guard_protects_long_crack():  # Req 3.6
    crack = _keep_map("crack")
    assert crack[1] is True and crack[2] is True  # long crack does not absorb the small crack
    spall = _keep_map("spall")
    assert spall[1] is True and spall[2] is False  # non-crack nested pair is deduplicated
