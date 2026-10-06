"""02_stream_client.py — N concurrent streaming requests; client-measured TTFT/ITL vs the server's own histograms.

Run:      uv run python course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/examples/02_stream_client.py \
            --url http://localhost:8001 --n 20 [--max-tokens 32] [--api-key KEY]
Needs:    a running OpenAI-compatible server (platform/mockllm, or vLLM in P2).
Expected: client p50 TTFT ≥ server p50 TTFT (the client also sees HTTP/connection time); ITLs agree closely.
Hardware: T0 (mock) / T2 (vLLM).
"""
import argparse
import asyncio
import json
import time

import httpx
import numpy as np


async def one(client: httpx.AsyncClient, url: str, max_tokens: int, headers: dict) -> tuple[float, list[float]]:
    body = {"model": "any", "prompt": "the quick brown fox " * 8, "max_tokens": max_tokens, "stream": True}
    t0 = time.perf_counter()
    stamps = []
    async with client.stream("POST", f"{url}/v1/completions", json=body, headers=headers, timeout=120) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if json.loads(line[6:])["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    return stamps[0] - t0, list(np.diff(stamps))


def server_quantile(metrics: str, name: str, q: float) -> float:
    """Prometheus-style interpolation over the scraped buckets (exercise 2 implements this properly)."""
    buckets = []
    for line in metrics.splitlines():
        if line.startswith(name + "_bucket"):
            le = line.split('le="')[1].split('"')[0]
            buckets.append((float("inf") if le == "+Inf" else float(le), float(line.rsplit(" ", 1)[1])))
    total = buckets[-1][1]
    rank, prev_le, prev_c = q * total, 0.0, 0.0
    for le, c in buckets:
        if c >= rank:
            return prev_le if le == float("inf") else prev_le + (le - prev_le) * (rank - prev_c) / max(c - prev_c, 1e-12)
        prev_le, prev_c = le, c
    return float("nan")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--api-key")
    a = ap.parse_args()
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    async with httpx.AsyncClient() as c:
        res = await asyncio.gather(*[one(c, a.url, a.max_tokens, headers) for _ in range(a.n)])
        metrics = (await c.get(f"{a.url}/metrics")).text
    ttft = np.array([r[0] for r in res]) * 1e3
    itl = np.concatenate([r[1] for r in res]) * 1e3
    print("| source | p50 TTFT ms | p90 TTFT ms | p50 ITL ms | p90 ITL ms |\n|---|---|---|---|---|")
    print(f"| client | {np.percentile(ttft, 50):.1f} | {np.percentile(ttft, 90):.1f} | {np.percentile(itl, 50):.1f} | {np.percentile(itl, 90):.1f} |")
    sq = lambda n, q: 1e3 * server_quantile(metrics, n, q)  # noqa: E731
    print(f"| server (/metrics, cumulative) | {sq('vllm:time_to_first_token_seconds', .5):.1f} | {sq('vllm:time_to_first_token_seconds', .9):.1f} "
          f"| {sq('vllm:inter_token_latency_seconds', .5):.1f} | {sq('vllm:inter_token_latency_seconds', .9):.1f} |")


if __name__ == "__main__":
    asyncio.run(main())
