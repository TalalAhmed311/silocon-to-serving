"""Timing helpers with the same contract as include/s2s/bench.hpp.

time_fn() warms up, repeats, and returns median / p90 in milliseconds.
save_results() writes results/<bench>.json in the shape the site charts read.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable


def time_fn(fn: Callable[[], object], warmup: int = 3, reps: int = 15) -> dict:
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1e3)
    ms.sort()
    p90 = ms[min(len(ms) - 1, round(0.9 * (len(ms) - 1)))]
    return {"median_ms": statistics.median(ms), "p90_ms": p90, "min_ms": ms[0]}


def hardware_string() -> str:
    return f"{platform.machine()} {platform.processor() or platform.system()}"


def save_results(bench: str, unit: str, rows: list[dict], outdir: str | Path = "results",
                 hardware: str | None = None) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{bench}.json"
    path.write_text(json.dumps({"bench": bench, "hardware": hardware or hardware_string(),
                                "unit": unit, "rows": rows}, indent=2))
    return path
