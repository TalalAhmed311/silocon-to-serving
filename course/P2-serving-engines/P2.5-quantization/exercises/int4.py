"""Exercise 1 starter."""
import numpy as np


def quantize(w: np.ndarray, group: int = 128, symmetric: bool = True):
    raise NotImplementedError


def dequantize(q, scale, zero, shape) -> np.ndarray:
    raise NotImplementedError


def pack(q: np.ndarray) -> np.ndarray:
    raise NotImplementedError  # optional (1b)


def unpack(packed: np.ndarray, n: int) -> np.ndarray:
    raise NotImplementedError  # optional (1b)
