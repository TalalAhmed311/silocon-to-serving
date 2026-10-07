"""Exercise 3: TP communication time per decode step. Contract: platform/capacity/core.py:tp_comm_seconds_per_step."""
from __future__ import annotations


def tp_comm_seconds_per_step(model, tp: int, tokens: int, link_gbs: float, alpha_us: float = 0.0, act_bytes: float = 2.0) -> float:
    """model has .layers and .hidden. Two ring all-reduces per layer of tokens × hidden × act_bytes bytes."""
    raise NotImplementedError("TODO")
