"""Exercise 2 starter. Every function except `column_copy` must return a VIEW: no new buffer, no bytes copied."""
import numpy as np


def row(a: np.ndarray, i: int) -> np.ndarray:
    """Row i of a 2-D array, as a view."""
    raise NotImplementedError


def column(a: np.ndarray, j: int) -> np.ndarray:
    """Column j of a 2-D array, as a view."""
    raise NotImplementedError


def transposed(a: np.ndarray) -> np.ndarray:
    """The transpose of a 2-D array, as a view."""
    raise NotImplementedError


def every_other_row(a: np.ndarray) -> np.ndarray:
    """Rows 0, 2, 4, ... of a 2-D array, as a view."""
    raise NotImplementedError


def column_copy(a: np.ndarray, j: int) -> np.ndarray:
    """Column j as an independent, contiguous copy. Writing to it must not change `a`."""
    raise NotImplementedError
