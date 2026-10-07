"""ncu_to_table.py — turn `ncu --csv --metrics ...` output into the D4 bench table with a "% of peak" column.

    sudo ncu --csv --metrics gpu__time_duration.sum,dram__bytes.sum,sm__sops_... ./binary > run.csv
    python platform/kernels/bench/ncu_to_table.py run.csv --peak-gbs <measured copy GB/s> [--peak-tflops <cited>]

Metrics used (Nsight Compute metric names; UNVERIFIED for your ncu version — `ncu --query-metrics` lists them):
  gpu__time_duration.sum               kernel time (ns, or as unit column says)
  dram__bytes.sum                      DRAM bytes moved (read + write)
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum   FFMA count (×2 FLOPs) — fp32 kernels only
ncu --csv emits one row per (kernel launch, metric) with columns incl. "Kernel Name", "Metric Name", "Metric Unit",
"Metric Value". Values may contain thousands separators.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict

UNIT = {"ns": 1e-9, "usecond": 1e-6, "us": 1e-6, "msecond": 1e-3, "ms": 1e-3, "second": 1.0, "s": 1.0,
        "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}


def parse(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]          # ncu prints ==PROF== lines too
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    per = defaultdict(dict)
    for r in rows:
        key = (r.get("ID", ""), r["Kernel Name"])
        val = float(r["Metric Value"].replace(",", "")) * UNIT.get(r.get("Metric Unit", ""), 1.0)
        per[key][r["Metric Name"]] = val
    out = []
    for (kid, name), m in per.items():
        t = m.get("gpu__time_duration.sum")
        if not t:
            continue
        row = {"id": kid, "kernel": name, "time_s": t}
        if "dram__bytes.sum" in m:
            row["gbs"] = m["dram__bytes.sum"] / t / 1e9
        ffma = m.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum")
        if ffma:
            row["tflops"] = 2 * ffma / t / 1e12
        out.append(row)
    return out


def table(rows: list[dict], peak_gbs: float | None, peak_tflops: float | None) -> str:
    def pct(x, peak):
        return f"{100 * x / peak:.0f}%" if (x is not None and peak) else "—"

    def num(x, fmt):
        return format(x, fmt) if x is not None else "—"

    lines = ["| kernel | time (µs) | GB/s | % of copy | TFLOP/s | % of peak |", "|---|---|---|---|---|---|"]
    for r in rows:
        g, f = r.get("gbs"), r.get("tflops")
        lines.append(f"| {r['kernel'][:50]} | {r['time_s'] * 1e6:.1f} | {num(g, '.1f')} | {pct(g, peak_gbs)} | "
                     f"{num(f, '.2f')} | {pct(f, peak_tflops)} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--peak-gbs", type=float)
    ap.add_argument("--peak-tflops", type=float)
    a = ap.parse_args(argv)
    rows = parse(open(a.csv).read())
    if not rows:
        print("no kernels with gpu__time_duration.sum found")
        return 1
    print(table(rows, a.peak_gbs, a.peak_tflops))
    return 0


if __name__ == "__main__":
    sys.exit(main())
