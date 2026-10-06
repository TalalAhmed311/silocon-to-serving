"""drain_test.py — keep N streams open against a URL for D seconds; count failed/truncated streams (P3.3 exercise 3).

Run: uv run python course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator/exercises/drain_test.py --url http://mockllm.s2s:8001
"""
import argparse
import asyncio
import json
import time

import httpx


async def worker(c, url, t_end, max_tokens, stats):
    while time.perf_counter() < t_end:
        got = 0
        try:
            async with c.stream("POST", f"{url}/v1/completions", timeout=60,
                                json={"prompt": "hi", "max_tokens": max_tokens, "stream": True}) as r:
                if r.status_code != 200:
                    stats["http_errors"] += 1
                    continue
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]" and json.loads(line[6:]).get("choices", [{}])[0].get("text"):
                        got += 1
            stats["ok" if got == max_tokens else "truncated"] += 1
        except httpx.HTTPError:
            stats["transport_errors"] += 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    stats = {"ok": 0, "truncated": 0, "http_errors": 0, "transport_errors": 0}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=None)) as c:
        t_end = time.perf_counter() + a.seconds
        await asyncio.gather(*[worker(c, a.url, t_end, a.max_tokens, stats) for _ in range(a.streams)])
    print(stats)
    raise SystemExit(0 if stats["truncated"] + stats["http_errors"] + stats["transport_errors"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
