"""Exercise 2 reference solution."""
import numpy as np


def row(a: np.ndarray, i: int) -> np.ndarray:
    return a[i]


def column(a: np.ndarray, j: int) -> np.ndarray:
    return a[:, j]


def transposed(a: np.ndarray) -> np.ndarray:
    return a.T


def every_other_row(a: np.ndarray) -> np.ndarray:
    return a[::2]


def column_copy(a: np.ndarray, j: int) -> np.ndarray:
    return np.ascontiguousarray(a[:, j])     # a[:, j].copy() works too
