"""Exercise 3 starter. No Python loops: use array expressions and broadcasting."""
import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """For a 2-D array, return (x - row_mean) / sqrt(row_variance + eps), row by row.

    Use the population variance (np.var with its default ddof=0). `eps` keeps a constant row finite.
    Hint: pass keepdims=True to mean and var so the results have shape (rows, 1) and broadcast against x.
    """
    raise NotImplementedError


def center_columns(x: np.ndarray) -> np.ndarray:
    """Subtract each column's mean from that column. Result has the same shape as x."""
    raise NotImplementedError
