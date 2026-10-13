"""Exercise 4 starter. `pairwise_distances_loop` is given and correct. Write the broadcast version."""
import numpy as np


def pairwise_distances_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d). Returns (n, m) where out[i, j] = Euclidean distance between a[i] and b[j]."""
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            diff = a[i] - b[j]
            out[i, j] = np.sqrt(np.sum(diff * diff))
    return out


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same result as the loop, with no Python loops.

    Hint: a[:, None, :] has shape (n, 1, d) and b[None, :, :] has shape (1, m, d).
    Subtracting them broadcasts to (n, m, d). Then reduce over the last axis.
    """
    raise NotImplementedError
