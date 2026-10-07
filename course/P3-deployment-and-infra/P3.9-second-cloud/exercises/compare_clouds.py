"""compare_clouds.py — the P3.9 cross-cloud table from your prices + #4 results, via the #2 cost model.

Input YAML (one entry per cloud; prices are YOURS, with date and source — the script refuses entries without them):
    - cloud: AWS
      node: g6.xlarge
      usd_per_hour: 0.0          # on-demand, your region
      price_source: "EC2 pricing page, us-east-1, 2026-10-01"
      gpus: 1
      loadgen: results/p3.6-ramp.json     # #4 output: knee tokens/s and p90 TTFT are read from it
      cold_start_s: 0             # from P3.6 coldstart.py
Usage: PYTHONPATH=platform python compare_clouds.py clouds.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from cost.model import Node, gpu_usd_per_hour, usd_per_million


def knee_from_loadgen(path: str) -> tuple[float, float]:
    """(output tokens/s at the knee, p90 TTFT there) from a loadgen.cli results file ({"rows", "knee"})."""
    d = json.loads(Path(path).read_text())
    k = d.get("knee")
    if not k:
        raise ValueError(f"{path}: no knee (the SLO failed even at the lowest rate) — rerun #4 with lower rates")
    return k["out_tok_s"], k["ttft_p90"]


def row(e: dict, util: float = 0.3) -> dict:
    for k in ("usd_per_hour", "price_source"):
        if not e.get(k):
            raise ValueError(f"{e.get('cloud')}: missing {k} — prices must be your own, dated and sourced")
    tok_s, ttft = (e["tok_s"], e.get("p90_ttft_s", float("nan"))) if "tok_s" in e else knee_from_loadgen(e["loadgen"])
    gpu_h = gpu_usd_per_hour(Node(e["node"], e["usd_per_hour"], e.get("gpus", 1)))
    full = usd_per_million(gpu_h, tok_s * 3600)
    return {"cloud": e["cloud"], "node": e["node"], "usd_gpu_h": gpu_h, "tok_s": tok_s, "p90_ttft_s": ttft,
            "usd_per_m_100": full, "usd_per_m_util": full / util, "cold_start_s": e.get("cold_start_s")}


def table(entries: list[dict], util: float = 0.3) -> str:
    rows = [row(e, util) for e in entries]
    out = ["| cloud | node | $/GPU-h | knee tok/s | p90 TTFT @ knee | $/1M @ 100% | $/1M @ %d%% | cold start s |" % (util * 100),
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['cloud']} | {r['node']} | {r['usd_gpu_h']:.3f} | {r['tok_s']:.0f} | {r['p90_ttft_s']:.2f} | "
                   f"{r['usd_per_m_100']:.3f} | {r['usd_per_m_util']:.3f} | {r['cold_start_s'] if r['cold_start_s'] is not None else '—'} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(table(yaml.safe_load(Path(sys.argv[1]).read_text())))
