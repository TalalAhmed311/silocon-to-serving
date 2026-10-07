"""decode_ceiling.py — measured decode tok/s at batch 1/2/4/8 vs the bandwidth ceiling from D3 (P1.5).

Run:      python course/P2-serving-engines/P2.1-first-serve-with-vllm/bench/decode_ceiling.py --url http://127.0.0.1:8000 \
            --gpu L4 --model-config /opt/models/<model>/config.json [--api-key K] [--weights bf16]
Output:   | batch | per-seq tok/s | aggregate tok/s | predicted ceiling per seq | % of ceiling |  + results/decode_ceiling.json
Method:   B concurrent streams, short prompt, 256 output tokens, temperature 0; per-seq tok/s = 1 / median ITL
          over the decode phase; 1 warm-up round discarded; median of 3 rounds.
Hardware: T2 (also runs against mockllm for a dry run).
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform"))
sys.path.insert(0, str(ROOT / "course/common/python"))
from capacity.core import Model, plan  # noqa: E402
from s2s.bench import save_results  # noqa: E402
from s2s.tables import md_table  # noqa: E402


async def stream(c, url, H, model, max_tokens):
    stamps = []
    async with c.stream("POST", f"{url}/v1/completions", headers=H, timeout=600,
                        json={"model": model, "prompt": "Count slowly:", "max_tokens": max_tokens, "temperature": 0,
                              "stream": True, "ignore_eos": True}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]" and json.loads(line[6:])["choices"][0].get("text"):
                stamps.append(time.perf_counter())
    return np.diff(stamps)


async def round_(url, H, model, B, max_tokens):
    async with httpx.AsyncClient() as c:
        gaps = await asyncio.gather(*[stream(c, url, H, model, max_tokens) for _ in range(B)])
    return float(np.median(np.concatenate(gaps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=os.environ.get("S2S_API_KEY"))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--weights", default="bf16")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    H = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    model = httpx.get(f"{a.url}/v1/models", headers=H).json()["data"][0]["id"]
    p = plan(Model.from_file(a.model_config), a.gpu, a.weights, a.weights, context=256, bw_util=1.0, batches=(1, 2, 4, 8))
    rows, js = [], []
    for B in (1, 2, 4, 8):
        asyncio.run(round_(a.url, H, model, B, 32))                     # warm-up
        itl = statistics.median(asyncio.run(round_(a.url, H, model, B, a.max_tokens)) for _ in range(3))
        per, agg = 1 / itl, B / itl
        ceil = p.decode_ceiling.get(B, (float("nan"),))[0]
        rows.append([B, per, agg, ceil, f"{100 * per / ceil:.0f}%"])
        js.append({"label": f"batch {B}", "size": B, "rate": per, "pct_peak": 100 * per / ceil, "aggregate": agg})
    print(f"model {model} on {a.gpu} [{p.gpu_status}]; ceiling uses 100% of spec bandwidth\n")
    print(md_table(["batch", "per-seq tok/s", "aggregate tok/s", "ceiling per seq", "% of ceiling"], rows))
    print("wrote", save_results("decode_ceiling", "tok/s", js, hardware=a.gpu))


if __name__ == "__main__":
    main()
