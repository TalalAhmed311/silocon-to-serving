"""parse_nccl_tests.py — nccl-tests stdout → `size | time us | algbw | busbw | % of link peak` (+ JSON).

Usage: python parse_nccl_tests.py results/nccl/all_reduce.txt --link-gbs 600 [--json results/nccl/all_reduce.json]
Reads the OUT-OF-PLACE columns. Each data row of nccl-tests ends with 8 fields:
time algbw busbw #wrong (out-of-place) time algbw busbw #wrong (in-place); size is the first field.
--link-gbs: the per-GPU unidirectional link bandwidth you are comparing against, from a cited spec (UNVERIFIED until you
cite it) — e.g. NVLink per-GPU bandwidth on p4d, or PCIe Gen4 x16 on a g6.12xlarge-class box.
"""
from __future__ import annotations

import argparse
import json
import sys


def parse(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        t = line.split()
        if len(t) < 9 or not t[0].isdigit() or line.lstrip().startswith("#"):
            continue
        try:
            rows.append({"size": int(t[0]), "time_us": float(t[-8]), "algbw": float(t[-7]), "busbw": float(t[-6])})
        except ValueError:
            continue
    return rows


def table(rows: list[dict], link_gbs: float | None) -> str:
    out = ["| size (B) | time (us) | algbw (GB/s) | busbw (GB/s) | % of link peak |", "|---|---|---|---|---|"]
    for r in rows:
        pct = f"{100 * r['busbw'] / link_gbs:.0f}%" if link_gbs else "—"
        out.append(f"| {r['size']} | {r['time_us']:.1f} | {r['algbw']:.2f} | {r['busbw']:.2f} | {pct} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--link-gbs", type=float)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = parse(open(a.file).read())
    if not rows:
        print("no data rows found")
        return 1
    print(table(rows, a.link_gbs))
    if a.json:
        open(a.json, "w").write(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
