"""report.py — render a reproducible benchmark report from a loadgen JSON; refuse if required metadata is missing.

Run:      uv run python course/P2-serving-engines/P2.4-benchmarking-methodology/examples/report.py results/lg.json \
            --hardware "1x L4 (g6.xlarge, us-east-1), driver 5xx, CUDA 12.x" --versions "vllm 0.31.0 (d1f3d8b8)" \
            --model "<repo>@<sha> bf16" --flags "--max-model-len 8192 --gpu-memory-utilization 0.9" [--repeats 3]
Hardware: T0.
"""
import argparse
import json
import sys
from pathlib import Path

REQUIRED = ["hardware", "versions", "model", "flags"]


def render(data: dict, meta: dict) -> str:
    missing = [k for k in REQUIRED if not meta.get(k)]
    if missing:
        raise ValueError("missing required report fields: " + ", ".join(missing))
    a = data["args"]
    lines = [f"# Benchmark report", "", "## Setup", "",
             f"- **Hardware:** {meta['hardware']}", f"- **Software:** {meta['versions']}", f"- **Model:** {meta['model']}",
             f"- **Engine flags:** `{meta['flags']}`",
             f"- **Load:** platform/loadgen, open-loop Poisson, seed {a.get('seed')}, {a.get('duration')} s per rate, "
             f"prompt mean {a.get('prompt_mean')} / output mean {a.get('output_mean')} (lognormal; word-count tokens)",
             f"- **SLO:** p90 TTFT ≤ {a.get('ttft_slo')} s, TPOT ≤ {a.get('tpot_slo')} s",
             f"- **Repeats:** {meta.get('repeats', 1)}" + (" (single run: treat as indicative)" if int(meta.get("repeats", 1)) < 3 else ""),
             "", "## Results", "", "| offered req/s | n | goodput req/s | out tok/s | p50 TTFT s | p90 TTFT s | p90 TPOT ms |",
             "|---|---|---|---|---|---|---|"]
    for r in data["rows"]:
        lines.append(f"| {r['offered_rps']:.2f} | {r['ok']} | {r['goodput_rps']:.2f} | {r['out_tok_s']:.0f} | "
                     f"{r['ttft_p50']:.2f} | {r['ttft_p90']:.2f} | {1e3 * r['tpot_p90']:.0f} |")
    k = data.get("knee")
    lines += ["", f"**Knee:** {k['offered_rps']:.2f} req/s" if k else "**Knee:** not reached", "",
              "## Raw data", "", f"`{meta.get('json_path', 'results/*.json')}` (commit it next to this report)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    for k in REQUIRED:
        ap.add_argument(f"--{k}")
    ap.add_argument("--repeats", default="1")
    a = ap.parse_args()
    try:
        print(render(json.loads(Path(a.json_path).read_text()), vars(a)))
    except ValueError as e:
        sys.exit(str(e))
