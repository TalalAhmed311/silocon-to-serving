"""mock_load.py — open-loop Poisson load at several request rates (or a fixed concurrency) against an OpenAI server.

Run:      uv run python course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/bench/mock_load.py \
            --url http://localhost:8001 --rates 5 20 50 100 [--duration 15] [--max-tokens 64]
          ... --concurrency 64      (closed-loop: keep exactly N requests in flight)
Output:   | rate | sent | p50 TTFT | p90 TTFT | p50 ITL | out tok/s | goodput (TTFT<1s, ITL p90<100ms) | + results/mock_load.json
Hardware: T0 (mock) / T2 (vLLM). This is the seed of #4 (P2.3), which adds ramps, prompt-length distributions and knee detection.
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "course/common/python"))
from s2s.bench import save_results  # noqa: E402
from s2s.tables import md_table  # noqa: E402


async def request(c, url, max_tokens, out):
    t0 = time.perf_counter()
    stamps = []
    try:
        async with c.stream("POST", f"{url}/v1/completions", timeout=300,
                            json={"model": "any", "prompt": "hello " * 64, "max_tokens": max_tokens, "stream": True}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]" and json.loads(line[6:])["choices"][0].get("text"):
                    stamps.append(time.perf_counter())
        out.append((stamps[0] - t0, np.diff(stamps), len(stamps)))
    except Exception as e:  # noqa: BLE001 — count failures, don't crash the run
        out.append(("error", str(e), 0))


async def run_rate(url, rate, duration, max_tokens, rng):
    out, tasks = [], []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None)) as c:
        t_end = time.perf_counter() + duration
        while time.perf_counter() < t_end:
            tasks.append(asyncio.create_task(request(c, url, max_tokens, out)))
            await asyncio.sleep(rng.expovariate(rate))   # open loop: arrivals don't wait for responses
        t_sent = time.perf_counter()
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - (t_end - duration)
    return out, len(tasks), wall


async def run_concurrency(url, n, duration, max_tokens):
    out = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None)) as c:
        t_end = time.perf_counter() + duration
        t0 = time.perf_counter()

        async def worker():
            while time.perf_counter() < t_end:
                await request(c, url, max_tokens, out)
        await asyncio.gather(*[worker() for _ in range(n)])
    return out, len(out), time.perf_counter() - t0


def summarize(label, out, sent, wall):
    ok = [o for o in out if o[0] != "error"]
    ttft = np.array([o[0] for o in ok])
    itl = np.concatenate([o[1] for o in ok]) if ok else np.array([np.nan])
    toks = sum(o[2] for o in ok)
    good = sum(1 for o in ok if o[0] < 1.0 and (len(o[1]) == 0 or np.percentile(o[1], 90) < 0.1))
    p = lambda v, q: float(np.percentile(v, q)) * 1e3 if len(v) else float("nan")  # noqa: E731
    return [label, sent, p(ttft, 50), p(ttft, 90), p(itl, 50), toks / wall, good / wall, len(out) - len(ok)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--rates", type=float, nargs="*", default=[5, 20, 50, 100])
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--duration", type=float, default=15)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    rows = []
    if a.concurrency:
        rows.append(summarize(f"{a.concurrency} in flight", *asyncio.run(run_concurrency(a.url, a.concurrency, a.duration, a.max_tokens))))
    else:
        rng = random.Random(0)
        for r in a.rates:
            rows.append(summarize(f"{r:g} req/s", *asyncio.run(run_rate(a.url, r, a.duration, a.max_tokens, rng))))
    hdr = ["load", "sent", "p50 TTFT ms", "p90 TTFT ms", "p50 ITL ms", "out tok/s", "goodput req/s", "errors"]
    print(md_table(hdr, rows))
    print("wrote", save_results("mock_load", "req/s", [dict(zip(hdr, r)) for r in rows]))


if __name__ == "__main__":
    main()
