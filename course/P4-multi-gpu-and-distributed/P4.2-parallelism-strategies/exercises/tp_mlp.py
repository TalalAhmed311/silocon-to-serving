"""Your tensor-parallel SwiGLU MLP (exercise 1). Same contract as parallel.tp_mlp:
return (list of per-rank partial outputs, their sum). Use a column split for w_gate/w_up, a row split for w_down."""
from __future__ import annotations

import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-x))


def tp_mlp(x, w_gate, w_up, w_down, tp: int):
    raise NotImplementedError("TODO: column-split gate/up, row-split down, return (partials, sum)")
