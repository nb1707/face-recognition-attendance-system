"""Strict, model-independent embedding validation; no IO or cloud imports."""
import numpy as np


def normalized_embedding(value, dimension, epsilon=1e-8):
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size != dimension or dimension < 1:
        raise ValueError(f"embedding_dimension: expected ({dimension},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("embedding_nonfinite")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= epsilon:
        raise ValueError("embedding_zero_norm")
    result = vector / norm
    result.setflags(write=False)
    return result
