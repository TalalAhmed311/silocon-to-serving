"""Exercise 1: your MIG layout planner. Contract: platform/partitioning/mig.py:plan — return a list of GPUs, each a
list of (tenant name, profile name), using as few GPUs as possible under the 7-compute / 8-memory slice limits."""
from __future__ import annotations

from partitioning.mig import COMPUTE, MEMORY, PROFILES, Tenant  # noqa: F401


def plan(gpu: str, tenants: list[Tenant]) -> list[list[tuple[str, str]]]:
    raise NotImplementedError("TODO: smallest fitting profile per tenant, then first-fit decreasing")
