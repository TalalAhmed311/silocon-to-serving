"""mig.py — MIG profiles and a layout planner (T0).

Profile table per the NVIDIA MIG User Guide (UNVERIFIED until checked against the guide for your driver; the guide also
lists placement rules that this planner simplifies). A GPU has 7 compute slices and 8 memory slices; a profile
"Ng.Mgb" uses N compute slices and a fixed number of memory slices.

Simplification: we check only the slice totals (≤ 7 compute, ≤ 8 memory) per GPU, not the exact start positions the
hardware allows. Real layouts must also be valid mig-parted configs: test the final layout with `nvidia-smi mig -cgi`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# name -> (compute slices, memory slices, memory GB)
PROFILES = {
    "A100-80GB": {"1g.10gb": (1, 1, 10), "2g.20gb": (2, 2, 20), "3g.40gb": (3, 4, 40), "4g.40gb": (4, 4, 40), "7g.80gb": (7, 8, 80)},
    "A100-40GB": {"1g.5gb": (1, 1, 5), "2g.10gb": (2, 2, 10), "3g.20gb": (3, 4, 20), "4g.20gb": (4, 4, 20), "7g.40gb": (7, 8, 40)},
    "H100-80GB": {"1g.10gb": (1, 1, 10), "1g.20gb": (1, 2, 20), "2g.20gb": (2, 2, 20), "3g.40gb": (3, 4, 40),
                  "4g.40gb": (4, 4, 40), "7g.80gb": (7, 8, 80)},
}
COMPUTE, MEMORY = 7, 8


@dataclass
class Tenant:
    name: str
    mem_gb: float                 # weights + KV budget + overhead the tenant's server needs
    min_compute: int = 1          # compute slices needed for its SLO (from a #11 measurement)


@dataclass
class Gpu:
    profiles: list = field(default_factory=list)      # (tenant, profile)
    compute: int = 0
    memory: int = 0


def smallest_profile(gpu: str, t: Tenant) -> str:
    fits = [(c, m, name) for name, (c, m, gb) in PROFILES[gpu].items() if gb >= t.mem_gb and c >= t.min_compute]
    if not fits:
        raise ValueError(f"{t.name} needs {t.mem_gb} GB / {t.min_compute} slices: no {gpu} MIG profile fits")
    return min(fits)[2]          # fewest compute slices, then fewest memory slices


def plan(gpu: str, tenants: list[Tenant]) -> list[Gpu]:
    """First-fit decreasing by (memory slices, compute slices). Returns the GPUs with their assigned profiles."""
    choices = [(t, smallest_profile(gpu, t)) for t in tenants]
    choices.sort(key=lambda tp: (PROFILES[gpu][tp[1]][1], PROFILES[gpu][tp[1]][0]), reverse=True)
    gpus: list[Gpu] = []
    for t, prof in choices:
        c, m, _ = PROFILES[gpu][prof]
        for g in gpus:
            if g.compute + c <= COMPUTE and g.memory + m <= MEMORY:
                break
        else:
            g = Gpu()
            gpus.append(g)
        g.profiles.append((t.name, prof))
        g.compute += c
        g.memory += m
    return gpus


def mig_parted_config(name: str, gpus: list[Gpu]) -> dict:
    """A mig-parted style config (the GPU Operator's MIG manager reads these from a ConfigMap) — one entry per GPU
    index. Schema per NVIDIA mig-parted README (UNVERIFIED for your operator version)."""
    entries = []
    for i, g in enumerate(gpus):
        counts: dict[str, int] = {}
        for _, prof in g.profiles:
            counts[prof] = counts.get(prof, 0) + 1
        entries.append({"devices": [i], "mig-enabled": True, "mig-devices": counts})
    return {"version": "v1", "mig-configs": {name: entries}}
