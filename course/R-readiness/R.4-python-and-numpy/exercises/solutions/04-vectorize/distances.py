"""Exercise 4 reference solution."""
import numpy as np


def pairwise_distances_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            diff = a[i] - b[j]
            out[i, j] = np.sqrt(np.sum(diff * diff))
    return out


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]          # (n, 1, d) - (1, m, d) -> (n, m, d)
    return np.sqrt(np.sum(diff * diff, axis=-1))  # reduce the last axis -> (n, m)
