"""commmodel.py — the P4.1 communication cost model, used by P4.2 (TP cost per token) and the capacity calculator.

alpha–beta model of a collective on n ranks moving S bytes per rank over links of bandwidth B (bytes/s), latency α (s):

    ring all-reduce      T = 2(n-1)·α + 2(n-1)/n · S/B
    ring reduce-scatter  T =  (n-1)·α +  (n-1)/n · S/B        (S = the full input size per rank)
    ring all-gather      T =  (n-1)·α +  (n-1)/n · S/B        (S = the full output size per rank)
    tree all-reduce      T ≈ 2·log2(n)·α + 2 · S/B            (latency-optimal; bandwidth-suboptimal)

nccl-tests reports algbw = S / t and busbw = algbw × factor, where the factor makes busbw comparable to link peak
(nccl-tests doc/PERFORMANCE.md at v2.21.1): all_reduce 2(n-1)/n · reduce_scatter (n-1)/n · all_gather (n-1)/n ·
broadcast 1 · reduce 1 · alltoall (n-1)/n · sendrecv 1.
"""
from __future__ import annotations

import math

BUSBW_FACTOR = {
    "all_reduce": lambda n: 2 * (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
    "sendrecv": lambda n: 1.0,
}


def busbw(collective: str, algbw: float, n: int) -> float:
    return algbw * BUSBW_FACTOR[collective](n)


def ring_allreduce_time(size_bytes: float, n: int, bw: float, alpha: float = 0.0) -> float:
    if n == 1:
        return 0.0
    return 2 * (n - 1) * alpha + 2 * (n - 1) / n * size_bytes / bw


def ring_rs_or_ag_time(size_bytes: float, n: int, bw: float, alpha: float = 0.0) -> float:
    if n == 1:
        return 0.0
    return (n - 1) * alpha + (n - 1) / n * size_bytes / bw


def tree_allreduce_time(size_bytes: float, n: int, bw: float, alpha: float = 0.0) -> float:
    if n == 1:
        return 0.0
    return 2 * math.log2(n) * alpha + 2 * size_bytes / bw


def fit_alpha_beta(sizes: list[float], times: list[float]) -> tuple[float, float]:
    """Least-squares fit of t = a + b·S. Returns (a, 1/b): intercept seconds and effective bandwidth bytes/s.
    For an all-reduce, divide a by 2(n-1) to get per-step α, and multiply 1/b by 2(n-1)/n to get the link bandwidth."""
    k = len(sizes)
    mx, my = sum(sizes) / k, sum(times) / k
    sxx = sum((x - mx) ** 2 for x in sizes)
    sxy = sum((x - mx) * (y - my) for x, y in zip(sizes, times))
    b = sxy / sxx
    a = my - b * mx
    return a, 1.0 / b


def crossover_size(n: int, bw: float, alpha: float) -> float:
    """Message size below which tree beats ring for all-reduce (both models above)."""
    # ring - tree = (2(n-1) - 2log2 n)·α + (2(n-1)/n - 2)·S/B = 0
    lat = (2 * (n - 1) - 2 * math.log2(n)) * alpha
    bwterm = (2 - 2 * (n - 1) / n) / bw
    return lat / bwterm
