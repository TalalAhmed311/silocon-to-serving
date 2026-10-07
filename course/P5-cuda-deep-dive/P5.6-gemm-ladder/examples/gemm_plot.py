"""gemm_plot.py — plot results/gemm.jsonl (written by gemm_ladder) as TFLOP/s vs N, one line per rung.

Run: uv run python course/P5-cuda-deep-dive/P5.6-gemm-ladder/examples/gemm_plot.py results/gemm.jsonl --out results/gemm.png
"""
import argparse
import json
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default="results/gemm.png")
    a = ap.parse_args()
    series = defaultdict(dict)
    for line in open(a.jsonl):
        d = json.loads(line)
        m = re.match(r"(.*) N=(\d+)$", d["label"])
        if m:
            series[m[1]][int(m[2])] = d["rate"]
    for name, pts in sorted(series.items()):
        xs = sorted(pts)
        plt.plot(xs, [pts[x] for x in xs], marker="o", label=name)
    plt.xscale("log", base=2)
    plt.xlabel("N (square matrices)")
    plt.ylabel("TFLOP/s")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
