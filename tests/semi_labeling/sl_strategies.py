"""Shared hypothesis strategies for semi-labeling property tests."""
from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp


def boxes(coord=st.floats(min_value=-50.0, max_value=300.0, allow_nan=False, allow_infinity=False)):
    """A box as (x1, y1, x2, y2); intentionally includes inverted / zero-area boxes."""
    return st.tuples(coord, coord, coord, coord)


def embedding_vectors(dim: int = 8, *, allow_zero: bool = True):
    """float32 embedding vectors; values bounded so cosine similarity is well defined."""
    elements = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32)
    vecs = hnp.arrays(dtype=np.float32, shape=dim, elements=elements)
    return vecs if allow_zero else vecs.filter(lambda v: float(np.linalg.norm(v)) > 1e-6)


def score_dicts(labels=("crack", "mold", "spall"), value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)):
    """A {label: score} mapping over a fixed label set, scores in [0, 1]."""
    return st.fixed_dictionaries({label: value for label in labels})
