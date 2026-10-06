"""Exercise 4 solution. Works in float64 so the identity is checked well below fp32 noise."""
import numpy as np


def _rope(x: np.ndarray, pos: int, theta: float) -> np.ndarray:
    hd = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.concatenate([pos * inv, pos * inv])
    half = hd // 2
    rot = np.concatenate([-x[half:], x[:half]])
    return x * np.cos(ang) + rot * np.sin(ang)


def score(q: np.ndarray, k: np.ndarray, m: int, n: int, theta: float = 10000.0) -> float:
    return float(_rope(q, m, theta) @ _rope(k, n, theta))


def score_via_offset(q: np.ndarray, k: np.ndarray, offset: int, theta: float = 10000.0) -> float:
    # R(m)ᵀR(n) = R(n − m) for 2-D rotations, applied pair by pair: rotate only k, by the offset.
    return float(q @ _rope(k, offset, theta))
