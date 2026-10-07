from __future__ import annotations

from partitioning import mig


def plan(gpu, tenants):
    return [g.profiles for g in mig.plan(gpu, tenants)]
