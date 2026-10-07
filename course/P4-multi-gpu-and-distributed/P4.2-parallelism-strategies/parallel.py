"""parallel.py — reference models for P4.2: tensor-parallel MLP numerics, pipeline schedules, memory per strategy.

Everything here is NumPy/pure Python (T0). The exercises re-implement parts of it; the tests compare against it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- tensor parallelism (Megatron MLP)
def silu(x):
    return x / (1.0 + np.exp(-x))


def mlp(x, w_gate, w_up, w_down):
    """SwiGLU MLP: (silu(x W_gate) * (x W_up)) W_down. x: [T, d]; w_gate, w_up: [d, f]; w_down: [f, d]."""
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def tp_mlp(x, w_gate, w_up, w_down, tp: int):
    """Column-split gate/up (each rank gets f/tp columns), row-split down (f/tp rows). Each rank computes a full-shape
    PARTIAL output; the all-reduce (sum) of the partials is the MLP output. The nonlinearity is elementwise on the
    f dimension, so it needs no communication — that is why the column-then-row order works."""
    f = w_gate.shape[1]
    assert f % tp == 0, "intermediate size must divide by tp"
    k = f // tp
    partials = []
    for r in range(tp):
        cols = slice(r * k, (r + 1) * k)
        h = silu(x @ w_gate[:, cols]) * (x @ w_up[:, cols])      # [T, f/tp] — local, no comm
        partials.append(h @ w_down[cols, :])                      # [T, d]  — partial sum
    return partials, np.sum(partials, axis=0)                     # the sum IS the all-reduce


def tp_mlp_wrong_order(x, w_gate, w_up, w_down, tp: int):
    """Row-split first (split d): silu would need the FULL x W_gate before it can be applied → a comm inside the MLP.
    Shown to make the point: summing silu of partial products is not silu of the sum."""
    d = w_gate.shape[0]
    k = d // tp
    parts = []
    for r in range(tp):
        rows = slice(r * k, (r + 1) * k)
        parts.append(silu(x[:, rows] @ w_gate[rows, :]) * (x[:, rows] @ w_up[rows, :]))
    return np.sum(parts, axis=0) @ w_down


# ---------------------------------------------------------------- pipeline parallelism
def bubble_fraction(stages: int, microbatches: int) -> float:
    """GPipe / 1F1B idle fraction with equal stage times: (p-1)/(m+p-1)."""
    return (stages - 1) / (microbatches + stages - 1)


@dataclass
class PipeResult:
    makespan: float
    busy: float
    bubble: float
    peak_inflight: int        # max microbatches whose activations a stage holds at once (memory!)


def simulate_pipeline(stages: int, microbatches: int, t_fwd: float = 1.0, t_bwd: float = 2.0,
                      schedule: str = "gpipe") -> PipeResult:
    """Event simulation of one training step. gpipe: all forwards, then all backwards. 1f1b: after a warm-up of
    (stages - s - 1) forwards, stage s alternates one forward and one backward."""
    p, m = stages, microbatches
    fin_f = [[0.0] * m for _ in range(p)]
    fin_b = [[0.0] * m for _ in range(p)]
    free = [0.0] * p
    if schedule == "gpipe":
        order = [[("F", j) for j in range(m)] + [("B", j) for j in reversed(range(m))] for _ in range(p)]
    elif schedule == "1f1b":
        order = []
        for s in range(p):
            warm = min(p - s - 1, m)
            o = [("F", j) for j in range(warm)]
            f, b = warm, 0
            while b < m:
                if f < m:
                    o.append(("F", f)); f += 1                       # noqa: E702
                o.append(("B", b)); b += 1                           # noqa: E702
            order.append(o)
    else:
        raise ValueError(schedule)
    # dependencies: F(s, j) after F(s-1, j); B(s, j) after B(s+1, j) and F(s, j). Iterate until all scheduled.
    idx = [0] * p
    done_f = [[False] * m for _ in range(p)]
    done_b = [[False] * m for _ in range(p)]
    inflight = [0] * p
    peak = 0
    remaining = sum(len(o) for o in order)
    while remaining:
        progressed = False
        for s in range(p):
            if idx[s] >= len(order[s]):
                continue
            kind, j = order[s][idx[s]]
            if kind == "F":
                if s > 0 and not done_f[s - 1][j]:
                    continue
                start = max(free[s], fin_f[s - 1][j] if s > 0 else 0.0)
                fin_f[s][j] = free[s] = start + t_fwd
                done_f[s][j] = True
                inflight[s] += 1
                peak = max(peak, inflight[s])
            else:
                if (s < p - 1 and not done_b[s + 1][j]) or not done_f[s][j]:
                    continue
                start = max(free[s], fin_b[s + 1][j] if s < p - 1 else fin_f[s][j])
                fin_b[s][j] = free[s] = start + t_bwd
                done_b[s][j] = True
                inflight[s] -= 1
            idx[s] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("deadlock in schedule")
    makespan = max(free)
    busy = m * (t_fwd + t_bwd)
    return PipeResult(makespan, busy, 1 - busy / makespan, peak)


# ---------------------------------------------------------------- memory per GPU for training (mixed precision Adam)
def training_bytes_per_param(strategy: str, n: int) -> float:
    """ZeRO paper accounting: 2 (bf16 params) + 2 (bf16 grads) + 12 (fp32 master + Adam m, v) = 16 B/param for DDP.
    zero1 shards optimizer state, zero2 also grads, zero3/fsdp also params."""
    if strategy == "ddp":
        return 16.0
    if strategy == "zero1":
        return 2 + 2 + 12 / n
    if strategy == "zero2":
        return 2 + (2 + 12) / n
    if strategy in ("zero3", "fsdp"):
        return 16.0 / n
    raise ValueError(strategy)
