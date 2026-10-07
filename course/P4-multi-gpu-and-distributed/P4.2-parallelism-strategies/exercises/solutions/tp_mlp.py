"""Reference for exercise 1 (same as ../../parallel.py:tp_mlp)."""
from __future__ import annotations

import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-x))


def tp_mlp(x, w_gate, w_up, w_down, tp: int):
    k = w_gate.shape[1] // tp
    partials = []
    for r in range(tp):
        c = slice(r * k, (r + 1) * k)
        partials.append((silu(x @ w_gate[:, c]) * (x @ w_up[:, c])) @ w_down[c, :])
    return partials, np.sum(partials, axis=0)
