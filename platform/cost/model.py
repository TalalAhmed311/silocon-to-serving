"""model.py — the cost model behind the #2 dashboard. Pure functions; every price is an INPUT (never hard-coded).

Core identity:  $/1M tokens = (hourly cost of the GPU share serving the traffic) / (tokens served per hour) × 1e6

Allocation rules (documented so finance and engineering agree):
  * a node's $/h is split across its GPUs equally; a MIG slice gets (slice compute fraction) × GPU $/h (P4.4)
  * a replica's cost is its GPU share's $/h; tenants are charged pro rata by tokens they consumed on that replica
  * idle capacity (provisioned but unused) is reported separately as "waste", not smeared into tenants' rates
  * spot: use the price actually paid (blended over the window), plus an interruption overhead factor for replays
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    name: str
    usd_per_hour: float          # what you pay: on-demand, reserved/savings-plan effective, or blended spot
    gpus: int
    spot: bool = False
    interruption_overhead: float = 0.0   # e.g. 0.03 = 3% of work redone after spot reclaims (measure it in P3.6)


@dataclass
class Replica:
    name: str
    node: Node
    gpus: float                  # GPUs this replica uses (1, 2 for TP=2, or a MIG fraction such as 1/7)
    tokens_per_hour: dict        # tenant -> tokens served in the window (prompt + completion, or completion only)


def gpu_usd_per_hour(node: Node) -> float:
    return node.usd_per_hour * (1 + node.interruption_overhead) / node.gpus


def replica_usd_per_hour(r: Replica) -> float:
    return gpu_usd_per_hour(r.node) * r.gpus


def usd_per_million(usd_per_hour: float, tokens_per_hour: float) -> float:
    return float("inf") if tokens_per_hour <= 0 else usd_per_hour / tokens_per_hour * 1e6


def tenant_costs(replicas: list[Replica]) -> dict:
    """Charge each tenant its token share of every replica it used. Returns tenant -> {"usd_per_hour", "tokens_per_hour",
    "usd_per_million"}. Replicas with zero traffic contribute only to waste()."""
    out: dict = {}
    for r in replicas:
        total = sum(r.tokens_per_hour.values())
        if total <= 0:
            continue
        cost = replica_usd_per_hour(r)
        for t, tok in r.tokens_per_hour.items():
            o = out.setdefault(t, {"usd_per_hour": 0.0, "tokens_per_hour": 0.0})
            o["usd_per_hour"] += cost * tok / total
            o["tokens_per_hour"] += tok
    for o in out.values():
        o["usd_per_million"] = usd_per_million(o["usd_per_hour"], o["tokens_per_hour"])
    return out


def waste(replicas: list[Replica], capacity_tokens_per_hour: dict) -> dict:
    """Idle cost per replica: (1 − utilization) × $/h, with utilization = served / measured capacity (from #4's knee)."""
    out = {}
    for r in replicas:
        cap = capacity_tokens_per_hour.get(r.name, 0)
        util = min(1.0, sum(r.tokens_per_hour.values()) / cap) if cap else 0.0
        out[r.name] = (1 - util) * replica_usd_per_hour(r)
    return out


def breakeven_utilization(gpu_usd_per_hour_: float, capacity_tokens_per_hour: float, api_usd_per_million: float) -> float:
    """Utilization at which self-hosting matches a per-token API price. Above it, self-hosting is cheaper."""
    full = usd_per_million(gpu_usd_per_hour_, capacity_tokens_per_hour)
    return full / api_usd_per_million
