"""fairness.py — Jain's fairness index and the #11 contention table (T0).

Jain(x) = (Σx)² / (n · Σx²): 1.0 when all tenants get the same throughput, 1/n when one tenant gets everything.
Normalise by each tenant's *entitlement* first (e.g. tok/s ÷ its share) when shares are unequal.
"""
from __future__ import annotations


def jain(xs: list[float]) -> float:
    xs = [float(x) for x in xs]
    s2 = sum(x * x for x in xs)
    return (sum(xs) ** 2) / (len(xs) * s2) if s2 else 1.0


def jain_weighted(xs: list[float], shares: list[float]) -> float:
    return jain([x / s for x, s in zip(xs, shares)])


def contention_row(mode: str, per_tenant_tok_s: list[float], p99_itl_ms: list[float]) -> str:
    return (f"| {mode} | {len(per_tenant_tok_s)} | " + " / ".join(f"{x:.0f}" for x in per_tenant_tok_s) +
            f" | {max(p99_itl_ms):.1f} | {jain(per_tenant_tok_s):.3f} |")
