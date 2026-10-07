"""Your scaling policy. Same API as platform/autoscaler/policy.py: Policy, State, raw_desired, step.

Starter: a stub that never changes the replica count. Replace raw_desired() and step().
"""
from __future__ import annotations

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
    history: list = field(default_factory=list)


def raw_desired(work: float, p: Policy) -> int:
    raise NotImplementedError("TODO: ceil(work / target_per_replica), 0 for no work")


def step(state: State, t: float, work: float, p: Policy) -> int:
    return state.replicas  # TODO
