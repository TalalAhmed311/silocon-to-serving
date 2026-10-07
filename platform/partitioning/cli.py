"""cli.py — #11 helpers.

    python -m partitioning.cli mig --gpu A100-80GB --tenant chat:35:3 --tenant embed:8 --tenant small:15:2
        → GPUs needed, per-GPU profiles, and a mig-parted config (YAML) to paste into k8s/mig-config.yaml
    python -m partitioning.cli table --mode mps results/p4.4/mps-4-t*.json
        → | mode | tenants | per-tenant tok/s | p99 ITL ms | Jain |   from #4 result files
"""
from __future__ import annotations

import argparse
import json

import yaml

from . import fairness, mig


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mig")
    m.add_argument("--gpu", default="A100-80GB", choices=sorted(mig.PROFILES))
    m.add_argument("--tenant", action="append", required=True, help="name:mem_gb[:min_compute_slices]")
    t = sub.add_parser("table")
    t.add_argument("--mode", required=True)
    t.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "mig":
        ts = []
        for spec in a.tenant:
            parts = spec.split(":")
            ts.append(mig.Tenant(parts[0], float(parts[1]), int(parts[2]) if len(parts) > 2 else 1))
        gpus = mig.plan(a.gpu, ts)
        print(f"{len(gpus)} × {a.gpu}")
        for i, g in enumerate(gpus):
            print(f"  GPU {i}: {g.profiles}  ({g.compute}/7 compute, {g.memory}/8 memory slices)")
        print(yaml.safe_dump(mig.mig_parted_config("s2s-mix", gpus), sort_keys=False))
    else:
        tok, itl = [], []
        for f in a.files:
            d = json.load(open(f))
            row = d["rows"][-1]
            tok.append(row.get("out_tok_s", 0.0))
            itl.append(row["itl_p99"] * 1e3)
        print("| mode | tenants | per-tenant tok/s | p99 ITL ms | Jain |\n|---|---|---|---|---|")
        print(fairness.contention_row(a.mode, tok, itl))


if __name__ == "__main__":
    main()
