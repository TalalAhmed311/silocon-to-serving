"""Exercise 3 reference solution."""
import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)          # shape (rows, 1)
    var = x.var(axis=1, keepdims=True)            # shape (rows, 1)
    return (x - mean) / np.sqrt(var + eps)        # (rows, cols) with (rows, 1): the 1 stretches


def center_columns(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0)                     # (rows, cols) with (cols,): aligned from the right
