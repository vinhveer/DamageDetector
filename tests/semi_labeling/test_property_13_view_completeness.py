# Feature: semi-labeling-pipeline-improvements, Property 13: View completeness
# For all successfully embedded detections, the set of persisted view_name values equals
# the set of configured views minus any views skipped for that detection.
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

import sl_imports

_ed = sl_imports.load_step("step3_embedding", "embed_detections")
embed_detection_multiview = _ed.embed_detection_multiview
VIEW_SPECS = _ed.VIEW_SPECS
_IMG = 100


class _MockEmbedder:
    dim = 8

    def embed(self, crops, *, batch_size):
        return np.zeros((len(crops), self.dim), dtype=np.float32)


_coord = st.floats(min_value=-20.0, max_value=120.0, allow_nan=False)


@settings(max_examples=60)  # each example does real (tiny) image IO
@given(x1=_coord, y1=_coord, x2=_coord, y2=_coord)
def test_view_completeness(x1, y1, x2, y2):
    configured = {name for name, _ in VIEW_SPECS}
    with tempfile.TemporaryDirectory() as d:
        Image.new("RGB", (_IMG, _IMG), (123, 222, 64)).save(os.path.join(d, "img.png"))
        det = SimpleNamespace(
            result_id=1, image_rel_path="img.png", image_path="img.png", source_input_dir=d,
            predicted_label="crack", x1=x1, y1=y1, x2=x2, y2=y2, image_width=_IMG, image_height=_IMG,
        )
        vectors, skips = embed_detection_multiview(det, None, _MockEmbedder(), VIEW_SPECS)

        skipped_views = {s.view_name for s in skips}
        assert set(vectors.keys()) == configured - skipped_views
        assert set(vectors.keys()).isdisjoint(skipped_views)
        assert set(vectors.keys()) | skipped_views == configured
        for vec in vectors.values():
            assert vec.shape[0] == _MockEmbedder.dim
