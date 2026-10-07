"""04_roofline.py — draw YOUR roofline from measured peak FLOP/s and bandwidth, and place measured kernels on it.

Run:      uv run python examples/04_roofline.py      (after 02_fma_peak, 03_stream and bench/sgemm_bench.py)
Output:   results/roofline.png and a printed table of ridge point and per-kernel bound.
Hardware: T0. Reads results/peak.json, results/stream.json, results/sgemm.json (if present).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("results")
peak = json.loads((R / "peak.json").read_text())["all_cores"]          # GFLOP/s
bw = json.loads((R / "stream.json").read_text())["triad_all_threads"]  # GB/s
ridge = peak / bw
print(f"P = {peak:.1f} GFLOP/s, B = {bw:.1f} GB/s, ridge I* = {ridge:.2f} FLOP/byte")

# Known kernels: (name, intensity FLOP/byte). Matmul uses n/6 (each matrix crosses DRAM once — the ideal).
kernels = [("vector add", 1 / 12), ("dot product", 0.25), ("matvec 4096²", 0.5), ("matmul n=256", 256 / 6),
           ("matmul n=2048", 2048 / 6)]
measured = []
if (R / "sgemm.json").exists():
    for row in json.loads((R / "sgemm.json").read_text())["rows"]:
        if row["label"] in ("simd", "simd+threads"):
            measured.append((f"D1 {row['label']} N={int(row['size'])}", row["size"] / 6, row["rate"]))

x = np.logspace(-2, 3, 400)
plt.figure(figsize=(8, 5))
plt.loglog(x, np.minimum(peak, x * bw), lw=2, label=f"roofline: min({peak:.0f}, I×{bw:.0f})")
plt.axvline(ridge, ls=":", c="gray")
plt.text(ridge * 1.1, peak * 0.3, f"ridge {ridge:.1f}", color="gray")
for name, i in kernels:
    y = min(peak, i * bw)
    plt.scatter([i], [y], s=25)
    plt.annotate(name, (i, y), textcoords="offset points", xytext=(5, -12), fontsize=8)
for name, i, rate in measured:
    plt.scatter([i], [rate], marker="x", s=60, c="red")
    plt.annotate(name, (i, rate), textcoords="offset points", xytext=(5, 5), fontsize=8, color="red")
plt.xlabel("arithmetic intensity (FLOP / DRAM byte)")
plt.ylabel("GFLOP/s")
plt.title("Your machine's roofline (fp32)")
plt.legend(loc="lower right")
plt.grid(True, which="both", alpha=0.3)
R.mkdir(exist_ok=True)
plt.savefig(R / "roofline.png", dpi=130, bbox_inches="tight")
print("| kernel | intensity | bound | ceiling GFLOP/s |\n|---|---|---|---|")
for name, i in kernels:
    print(f"| {name} | {i:.3g} | {'memory' if i < ridge else 'compute'} | {min(peak, i * bw):.1f} |")
print(f"wrote {R / 'roofline.png'}")
