"""budgets.py — per-tenant token budgets that survive active-active serving (C1 exercise 4).

Each region's gateway counts the tokens each tenant used there. A grow-only counter per (tenant, region) — a G-counter
CRDT — merges by element-wise max, so replicas can exchange state in any order, any number of times, and converge.
A tenant's usage is the sum over regions. Between syncs, each region only sees its own increments: a tenant can
overspend by at most (regions − 1) × (its spend rate × sync interval). `admit()` makes that bound explicit by
reserving a per-region slice of the remaining budget (`local_share`).
"""
from __future__ import annotations

from collections import defaultdict


class GCounter:
    def __init__(self, region: str):
        self.region = region
        self.c: dict[tuple[str, str], int] = defaultdict(int)     # (tenant, region) → tokens

    def add(self, tenant: str, n: int) -> None:
        assert n >= 0, "grow-only"
        self.c[(tenant, self.region)] += n

    def merge(self, other: "GCounter") -> None:
        for k, v in other.c.items():
            if v > self.c[k]:
                self.c[k] = v

    def used(self, tenant: str) -> int:
        return sum(v for (t, _), v in self.c.items() if t == tenant)

    def state(self) -> dict:
        return {f"{t}|{r}": v for (t, r), v in self.c.items()}


class RegionBudget:
    """One region's admission gate for per-tenant budgets.

    naive (split=False): admit if this region's VIEW of usage + n ≤ budget — every region can spend the whole
        remainder concurrently, so the tenant can overspend by up to (regions − 1) × remaining between syncs.
    split (split=True): at each sync, this region gets allowance = remaining / regions and spends only that until the
        next sync. Total spend between syncs ≤ remaining, so the budget is never exceeded — at the cost of refusing
        requests early when traffic is skewed toward one region (the exercise measures that refusal rate).
    """

    def __init__(self, counter: GCounter, budgets: dict[str, int], regions: int, split: bool = True):
        self.counter, self.budgets, self.regions, self.split = counter, budgets, regions, split
        self.allowance: dict[str, float] = {}
        self.spent: dict[str, int] = defaultdict(int)
        self.on_sync()

    def on_sync(self) -> None:
        """Call after merging the other regions' counters."""
        for t, b in self.budgets.items():
            self.allowance[t] = max(0, b - self.counter.used(t)) / self.regions
            self.spent[t] = 0

    def admit(self, tenant: str, n: int) -> bool:
        if self.split:
            ok = self.spent[tenant] + n <= self.allowance[tenant]
        else:
            ok = self.counter.used(tenant) + n <= self.budgets[tenant]
        if ok:
            self.spent[tenant] += n
            self.counter.add(tenant, n)
        return ok
