"""Exercise 4 starter."""
import numpy as np


def score(q: np.ndarray, k: np.ndarray, m: int, n: int, theta: float = 10000.0) -> float:
    raise NotImplementedError  # TODO: <RoPE(q, m), RoPE(k, n)>


def score_via_offset(q: np.ndarray, k: np.ndarray, offset: int, theta: float = 10000.0) -> float:
    raise NotImplementedError  # TODO: <q, RoPE(k, offset)>  — no absolute positions
