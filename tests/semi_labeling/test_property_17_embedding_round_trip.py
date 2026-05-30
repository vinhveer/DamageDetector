# Feature: semi-labeling-pipeline-improvements, Property 17: Embedding encoding round-trip
# For all produced embeddings, the stored vector dimension equals the configured model
# dimension, the stored byte length equals dim x element size, and decoding the stored
# blob reproduces the vector.
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import sl_imports

_os = sl_imports.load_step("step3_embedding", "output_store")
encode_vector = _os.encode_vector
decode_vector = _os.decode_vector

_ELEMENT_SIZE = np.dtype("<f4").itemsize  # 4 bytes


@given(
    vec=st.integers(min_value=1, max_value=64).flatmap(
        lambda d: hnp.arrays(
            dtype=np.float32, shape=d,
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=32),
        )
    )
)
def test_embedding_encoding_round_trip(vec):
    dim = int(vec.shape[0])
    blob = encode_vector(vec)
    assert len(blob) == dim * _ELEMENT_SIZE
    decoded = decode_vector(blob, dim)
    assert decoded.shape[0] == dim
    assert np.array_equal(decoded, vec.astype("<f4"))
