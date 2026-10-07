"""table.py — build the #7 comparison table from results/bakeoff/*.json (T0; runs anywhere once you copied results back).

Run:  python platform/bakeoff/table.py results/bakeoff
Columns: variant | VRAM MiB | KV tokens | p50 TTFT @ low rate | decode tok/s @ low rate | knee req/s | <task scores> | Δ vs baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def task_scores(eval_dir: str | None) -> dict:
    if not eval_dir or not Path(eval_dir).exists():
        return {}
    files = sorted(Path(eval_dir).rglob("results*.json"))
    if not files:
        return {}
    res = json.loads(files[-1].read_text()).get("results", {})
    out = {}
    for task, metrics in res.items():
        for k in ("exact_match,strict-match", "acc_norm,none", "acc,none", "exact_match,flexible-extract"):
            if k in metrics:
                out[task] = metrics[k]
                break
    return out


def build(results_dir: Path, baseline: str = "bf16") -> tuple[list[str], list[list]]:
    recs = {r["variant"]: r for r in (json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))
                                      if not p.name.endswith("_load.json"))}
    tasks = sorted({t for r in recs.values() for t in task_scores(r.get("eval_dir"))})
    base = task_scores(recs.get(baseline, {}).get("eval_dir"))
    hdr = ["variant", "VRAM MiB", "KV tokens", "p50 TTFT s (lowest rate)", "decode tok/s/seq (lowest rate)", "knee req/s"] + \
          [f"{t}" for t in tasks] + [f"Δ{t} vs {baseline}" for t in tasks]
    rows = []
    for name, r in recs.items():
        lo = r["load"]["rows"][0] if r.get("load") else {}
        knee = (r.get("load") or {}).get("knee")
        sc = task_scores(r.get("eval_dir"))
        tpot = lo.get("tpot_p50")
        rows.append([name, r.get("vram_used_mib"), r.get("kv_tokens"), lo.get("ttft_p50"), 1 / tpot if tpot else None,
                     knee["offered_rps"] if knee else None] + [sc.get(t) for t in tasks] +
                    [None if t not in sc or t not in base else sc[t] - base[t] for t in tasks])
    return hdr, rows


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "course/common/python"))
    from s2s.tables import md_table
    hdr, rows = build(Path(sys.argv[1] if len(sys.argv) > 1 else "results/bakeoff"))
    print(md_table(hdr, [[("—" if v is None else v) for v in r] for r in rows]))
