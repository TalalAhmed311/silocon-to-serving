"""policy.py — the #3 scaling decision as a pure function (what KEDA+HPA compute, made explicit and testable).

desired = clamp(ceil(work / target_per_replica), min, max), with:
  * scale-from-zero when any work is waiting (activation)
  * scale-to-zero only after `cooldown_s` with zero work
  * scale-down limited to `max_down_per_step` and only after `down_stabilization_s` of consistently lower demand
    (GPU capacity is expensive to re-acquire: flapping costs cold starts)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Policy:
    target_per_replica: float = 24
    min_replicas: int = 0
    max_replicas: int = 4
    cooldown_s: float = 600
    down_stabilization_s: float = 300
    max_up_per_step: int = 2
    max_down_per_step: int = 1


@dataclass
class State:
    replicas: int = 0
    last_nonzero_work_t: float = -1e18
    history: list = field(default_factory=list)   # (t, raw desired) for stabilization


def raw_desired(work: float, p: Policy) -> int:
    return 0 if work <= 0 else math.ceil(work / p.target_per_replica)


def step(state: State, t: float, work: float, p: Policy) -> int:
    if work > 0:
        state.last_nonzero_work_t = t
    want = raw_desired(work, p)
    state.history = [(ht, hd) for ht, hd in state.history if t - ht <= p.down_stabilization_s] + [(t, want)]
    if want > state.replicas:                                   # scale up immediately (bounded rate)
        new = min(want, state.replicas + p.max_up_per_step)
    else:                                                       # scale down: the MAX desired over the window
        stable = max(hd for _, hd in state.history)
        new = max(stable, state.replicas - p.max_down_per_step) if stable < state.replicas else state.replicas
    if new == 0 and t - state.last_nonzero_work_t < p.cooldown_s:
        new = max(1, min(state.replicas, 1))                    # keep one warm replica until the cooldown passes
    new = max(p.min_replicas, min(p.max_replicas, new))
    state.replicas = new
    return new
