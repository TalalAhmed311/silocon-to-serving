"""Check the S3 concurrency curve in results/p3.7-loader.json (rows named 's3-download c=N')."""
from __future__ import annotations

import argparse
import json
import re
import sys


def curve(rows):
    out = []
    for r in rows:
        m = re.fullmatch(r"s3-download c=(\d+)", r["method"])
        if m:
            out.append((int(m[1]), r["gb"] / r["seconds"]))
    return sorted(out)


def analyze(pts):
    best = max(g for _, g in pts)
    knee = next(c for c, g in pts if g >= 0.9 * best)
    return {"single": pts[0][1], "best": best, "knee": knee, "speedup": best / pts[0][1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--instance-gbps", type=float)
    a = ap.parse_args(argv)
    pts = curve(json.load(open(a.results)))
    if len(pts) < 3 or pts[0][0] != 1:
        print("FAIL: need at least 3 concurrency points including c=1")
        return 1
    print("| concurrency | GB/s |\n|---|---|")
    for c, g in pts:
        print(f"| {c} | {g:.2f} |")
    s = analyze(pts)
    print(f"knee at c={s['knee']}, best {s['best']:.2f} GB/s = {s['speedup']:.1f}× single stream")
    if a.instance_gbps:
        print(f"best = {100 * s['best'] * 8 / a.instance_gbps:.0f}% of the instance's {a.instance_gbps} Gbps")
    if s["speedup"] < 2:
        print("FAIL: concurrency gave < 2× — find the real bottleneck (disk?)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
