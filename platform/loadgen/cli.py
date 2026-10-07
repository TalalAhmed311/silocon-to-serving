"""cli.py — #4: ramp the request rate until the server's latency knee, print a table, save JSON for plotting.

Run:  PYTHONPATH=platform uv run python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 8 16 32 \
        --duration 60 --ttft-slo 2.0 --tpot-slo 0.1 [--prompt-mean 512 --output-mean 128] [--shared-prefix 1000 --shared-fraction 0.8]
      Watch the server's /metrics alongside (vllm:kv_cache_usage_perc, vllm:num_requests_waiting, vllm:num_preemptions).
Output: Markdown table per rate + the knee, and results/loadgen.json.
Hardware: T0 against mockllm; T2 against vLLM/SGLang.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from .analysis import find_knee, summarize
from .runner import run
from .workload import Workload


def scrape(url: str, headers: dict, names=("vllm:kv_cache_usage_perc", "vllm:num_requests_waiting", "vllm:num_preemptions_total")) -> dict:
    try:
        text = httpx.get(f"{url}/metrics", headers=headers, timeout=5).text
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for line in text.splitlines():
        for n in names:
            if line.startswith(n + " ") or line.startswith(n + "{"):
                out[n] = out.get(n, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s2s-loadgen")
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key", default=os.environ.get("S2S_API_KEY"))
    ap.add_argument("--rates", type=float, nargs="+", required=True)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--ttft-slo", type=float, default=2.0)
    ap.add_argument("--tpot-slo", type=float, default=0.1)
    ap.add_argument("--prompt-mean", type=int, default=512)
    ap.add_argument("--output-mean", type=int, default=128)
    ap.add_argument("--shared-prefix", type=int, default=0)
    ap.add_argument("--shared-fraction", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/loadgen.json")
    a = ap.parse_args(argv)
    H = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    model = httpx.get(f"{a.url}/v1/models", headers=H).json()["data"][0]["id"]
    rows = []
    print("| offered req/s | goodput req/s | out tok/s | p50 TTFT s | p90 TTFT s | p90 TPOT ms | errors | KV usage | waiting | preemptions |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for rate in sorted(a.rates):
        w = Workload(rate=rate, duration_s=a.duration, prompt_mean=a.prompt_mean, output_mean=a.output_mean,
                     shared_prefix_tokens=a.shared_prefix, shared_fraction=a.shared_fraction, seed=a.seed)
        before = scrape(a.url, H)
        res = asyncio.run(run(w, a.url, model, a.api_key))
        after = scrape(a.url, H)
        s = summarize(res, a.duration, a.ttft_slo, a.tpot_slo)
        s.update(rate=rate, kv_usage=after.get("vllm:kv_cache_usage_perc"), waiting=after.get("vllm:num_requests_waiting"),
                 preemptions=after.get("vllm:num_preemptions_total", 0) - before.get("vllm:num_preemptions_total", 0))
        rows.append(s)
        f = lambda v, k=1: "—" if v is None else f"{v:.{k}f}"  # noqa: E731
        print(f"| {rate:g} | {s['goodput_rps']:.2f} | {s['out_tok_s']:.0f} | {s['ttft_p50']:.2f} | {s['ttft_p90']:.2f} | "
              f"{1e3 * s['tpot_p90']:.0f} | {s['errors']} | {f(s['kv_usage'], 2)} | {f(s['waiting'], 0)} | {f(s['preemptions'], 0)} |")
    knee = find_knee(rows, a.ttft_slo)
    print(f"\nknee (p90 TTFT ≤ {a.ttft_slo}s and goodput ≥ 90% of offered): "
          + (f"{knee['offered_rps']:.2f} req/s, {knee['out_tok_s']:.0f} out tok/s" if knee else "not reached at the lowest rate"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows, "knee": knee}, indent=2, default=str))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
