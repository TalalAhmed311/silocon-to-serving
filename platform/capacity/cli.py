"""cli.py — s2s-capacity: print a capacity plan for model × GPU.

Run:  uv run python -m capacity.cli --model llama3-8b --gpu L4                    (from repo root, with platform on path:
      PYTHONPATH=platform uv run python -m capacity.cli --model llama3-8b --gpu L4 --weights fp8 --kv fp8 --context 8192)
      --model accepts a preset name (platform/capacity/presets/*.json) or a path to any config.json.
Hardware: T0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import Model, plan, presets_dir


def load_model(arg: str) -> Model:
    p = Path(arg)
    if not p.exists():
        p = presets_dir() / f"{arg}.json"
    m = Model.from_file(p)
    m.name = p.stem if p.parent == presets_dir() else m.name
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s2s-capacity")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--weights", default="bf16")
    ap.add_argument("--kv", default="bf16")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--mem-util", type=float, default=0.90)
    ap.add_argument("--bw-util", type=float, default=0.80)
    ap.add_argument("--mfu", type=float, default=0.50)
    a = ap.parse_args(argv)
    m = load_model(a.model)
    p = plan(m, a.gpu, a.weights, a.kv, a.tp, a.context, a.mem_util, bw_util=a.bw_util, mfu=a.mfu)
    print(f"# {p.model} on {a.tp}× {p.gpu}  [{p.gpu_status} GPU spec]\n")
    print("| quantity | value |\n|---|---|")
    print(f"| total params | {m.total_params() / 1e9:.2f} B (active per token: {m.active_params() / 1e9:.2f} B) |")
    print(f"| weights per GPU ({p.weight_dtype}) | {p.weights_gb_per_gpu:.1f} GiB |")
    print(f"| KV budget per GPU | {p.kv_budget_gb_per_gpu:.1f} GiB |")
    print(f"| KV per token ({p.kv_dtype}) | {p.kv_bytes_per_token / 1024:.0f} KiB |")
    print(f"| max KV tokens | {p.max_kv_tokens:,} |")
    print(f"| max sequences at {p.context} ctx | {p.max_seqs_at_context} |")
    if not p.fits:
        print("\n**DOES NOT FIT**: weights + activations exceed usable memory. Raise --tp, quantize, or use a bigger GPU.")
        return 1
    print("\n| batch | per-seq decode tok/s (ceiling) | aggregate tok/s |\n|---|---|---|")
    for B, (per, agg) in p.decode_ceiling.items():
        print(f"| {B} | {per:.1f} | {agg:.0f} |")
    if p.prefill_s:
        print("\n| prompt tokens | prefill (TTFT floor) |\n|---|---|")
        for T, s in p.prefill_s.items():
            print(f"| {T} | {s * 1e3:.0f} ms |")
    print(f"\nassumptions: {p.assumptions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
