"""01_closed_vs_open.py — coordinated omission, demonstrated: closed-loop vs open-loop at the same mean rate.

Run:      uv run python course/P2-serving-engines/P2.4-benchmarking-methodology/examples/01_closed_vs_open.py --url http://127.0.0.1:8001
Method:   1) closed loop with C users for D seconds -> achieved rate λ; 2) open-loop Poisson at λ for D seconds.
Output:   | load model | achieved req/s | p50 TTFT | p99 TTFT | n |   Hardware: T0 (mock) / T2.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform"))
import httpx  # noqa: E402

from loadgen.runner import Result, _one, run  # noqa: E402
from loadgen.workload import Request, Workload  # noqa: E402


async def closed_loop(url, users, duration, out_tokens):
    results = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None)) as c:
        t_end = time.perf_counter() + duration

        async def user():
            while time.perf_counter() < t_end:
                r = Result(Request(0.0, 64, out_tokens))
                await _one(c, url, "any", {}, r, 0, time.perf_counter(), 300)   # send now, wait for completion
                results.append(r)
        await asyncio.gather(*[user() for _ in range(users)])
    return results


def stats(results, duration):
    t = np.array([r.ttft for r in results if r.ttft is not None])
    return len(results) / duration, np.percentile(t, 50), np.percentile(t, 99), len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--users", type=int, default=32)
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--out-tokens", type=int, default=64)
    a = ap.parse_args()
    closed = asyncio.run(closed_loop(a.url, a.users, a.duration, a.out_tokens))
    lam, *cstats = stats(closed, a.duration)
    w = Workload(rate=lam, duration_s=a.duration, prompt_mean=64, prompt_sigma=0.01, output_mean=a.out_tokens, output_sigma=0.01)
    opened = asyncio.run(run(w, a.url))
    print("| load model | achieved req/s | p50 TTFT ms | p99 TTFT ms | n |\n|---|---|---|---|---|")
    print(f"| closed loop, {a.users} users | {lam:.1f} | {1e3 * cstats[0]:.0f} | {1e3 * cstats[1]:.0f} | {cstats[2]} |")
    o = stats(opened, a.duration)
    print(f"| open loop, Poisson at {lam:.1f}/s | {o[0]:.1f} | {1e3 * o[1]:.0f} | {1e3 * o[2]:.0f} | {o[3]} |")
    print("\nThe closed loop can never queue more than its user count; Poisson bursts can — that tail is real.")


if __name__ == "__main__":
    main()
