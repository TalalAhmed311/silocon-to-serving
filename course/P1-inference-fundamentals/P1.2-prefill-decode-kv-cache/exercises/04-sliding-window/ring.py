"""Exercise 4 starter."""
import numpy as np


class RingKV:
    def __init__(self, window: int, layers: int, kv_dim: int):
        raise NotImplementedError

    def write(self, layer: int, pos: int, k: np.ndarray, v: np.ndarray) -> None:
        raise NotImplementedError

    def read(self, layer: int, pos: int):
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError


def windowed_attention(q, K, V, n_heads, n_kv_heads, hd):
    raise NotImplementedError
