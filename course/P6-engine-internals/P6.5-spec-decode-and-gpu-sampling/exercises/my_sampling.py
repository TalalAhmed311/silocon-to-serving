"""Your sampler pieces. References: platform/engine/v1/s2s_engine/sampling.py and spec_verify.py."""
from __future__ import annotations

import numpy as np


def top_p_threshold(probs: np.ndarray, top_p: float, iters: int = 40) -> float:
    """TODO (ex. 2): the largest τ with probs[probs >= τ].sum() >= top_p, by bisection on [0, max(probs)]. No sort."""
    raise NotImplementedError


def verify_batch(draft: np.ndarray, q: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> list[list[int]]:
    """TODO (ex. 1): draft [B, k]; q [B, k, V]; p [B, k+1, V] → per sequence: accepted prefix + one corrected or
    bonus token (README §2)."""
    raise NotImplementedError
