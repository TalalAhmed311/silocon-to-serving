"""faults.py — checkpoint-interval math and a failure simulator (T0).

Young/Daly: with checkpoint cost C (seconds the job is paused to save) and mean time between failures M, the interval
that minimises expected wasted time is approximately  τ* = sqrt(2·C·M)  (Young 1974; Daly 2006 adds higher-order
terms: τ* = sqrt(2CM)·(1 + sqrt(C/2M)/3 + C/(18M)) − C for C < 2M).

Waste per unit time ≈ C/τ (saving) + τ/(2M) (expected recomputation after a failure) + R/M (restart cost R).
"""
from __future__ import annotations

import math
import random


def young(C: float, M: float) -> float:
    return math.sqrt(2 * C * M)


def daly(C: float, M: float) -> float:
    if C >= 2 * M:
        return M
    return math.sqrt(2 * C * M) * (1 + math.sqrt(C / (2 * M)) / 3 + C / (18 * M)) - C


def waste_fraction(tau: float, C: float, M: float, R: float = 0.0) -> float:
    """First-order expected fraction of wall-clock not spent on useful work."""
    return C / (tau + C) + (tau + C) / (2 * M) + R / M


def simulate(work: float, tau: float, C: float, M: float, R: float = 0.0, seed: int = 0, trials: int = 200) -> float:
    """Monte-Carlo mean wall-clock to finish `work` seconds of compute with a checkpoint every `tau` seconds of compute
    (each costing C), exponential failures with mean M, restart cost R. Failures can strike during saves too."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(trials):
        t, done = 0.0, 0.0
        next_fail = rng.expovariate(1 / M)
        while done < work:
            seg = min(tau, work - done)
            need = seg + (C if done + seg < work else 0.0)          # no checkpoint after the final segment
            if t + need <= next_fail:
                t += need
                done += seg
            else:                                                   # lose the partial segment, restart, reload
                t = next_fail + R
                next_fail = t + rng.expovariate(1 / M)
        total += t
    return total / trials
