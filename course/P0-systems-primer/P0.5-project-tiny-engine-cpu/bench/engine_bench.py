"""engine_bench.py — decode tok/s of #0 v0 vs the bandwidth-bound ceiling B / W.

Run:      uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/bench/engine_bench.py --model build/tiny-llama
          [--engine build/engine-v0/s2s-engine] [--threads 1 2 4 8] [--stream course/.../P0.4.../results/stream.json]
Output:   | model | threads | prompt toks | prefill tok/s | decode tok/s | predicted ceiling | % of ceiling |
          + results/engine.json. Each configuration runs 5 times; the table reports the median.
Hardware: T0.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "course" / "common" / "python"))
from s2s.bench import save_results  # noqa: E402
from s2s.tables import md_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engine", default=str(ROOT / "build" / "engine-v0" / "s2s-engine"))
ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--stream", default=str(ROOT / "course/P0-systems-primer/P0.4-simd-and-roofline/results/stream.json"))
a = ap.parse_args()

bw = None
if Path(a.stream).exists():
    bw = json.loads(Path(a.stream).read_text())["triad_all_threads"] * 1e9
else:
    print(f"(no {a.stream}: run P0.4 examples/03_stream to get B; ceiling column will be blank)")

pat = re.compile(r"weights ([\d.]+) MiB.*prefill (\d+) tok in [\d.]+ s \(([\d.]+) tok/s\).*decode \d+ tok in [\d.]+ s \(([\d.]+) tok/s\)")
rows, out = [], []
prompt = " ".join(str(i % 50 + 1) for i in range(16))
for t in a.threads:
    pre, dec, wmib = [], [], 0.0
    for _ in range(5):
        r = subprocess.run([a.engine, "--model", a.model, "--prompt-ids", prompt, "--steps", str(a.steps), "--threads", str(t)],
                           capture_output=True, text=True, check=True)
        m = pat.search(r.stderr)
        wmib, pre_t, dec_t = float(m.group(1)), float(m.group(3)), float(m.group(4))
        pre.append(pre_t)
        dec.append(dec_t)
    ceiling = bw / (wmib * 2**20) if bw else None
    d = statistics.median(dec)
    rows.append([Path(a.model).name, t, 16, statistics.median(pre), d, ceiling or "", f"{100 * d / ceiling:.0f}%" if ceiling else ""])
    out.append({"label": f"{t} threads", "size": t, "rate": d, "pct_peak": 100 * d / ceiling if ceiling else 0})
print(md_table(["model", "threads", "prompt toks", "prefill tok/s", "decode tok/s", "predicted ceiling (B/W)", "% of ceiling"], rows))
print("wrote", save_results("engine", "tok/s", out))
