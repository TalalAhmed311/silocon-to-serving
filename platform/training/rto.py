"""rto.py — RTO/RPO from #9 logs: a run that died (injected or real) and the run that resumed.

    python -m training.rto results/p4.3/run1.log results/p4.3/run2.log --restart-s <seconds between crash and relaunch>
RPO (work lost)  = failed_step - resumed_from - 1 steps (× seconds per step)
RTO (time to recover) = restart delay + time from relaunch to the first new step logged
"""
from __future__ import annotations

import argparse
import json


def events(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def analyze(run1: list[dict], run2: list[dict], restart_s: float) -> dict:
    fail = next(e for e in run1 if e.get("event") == "injected_failure")
    steps1 = [e for e in run1 if "loss" in e]
    sec_per_step = (steps1[-1]["t"] - steps1[0]["t"]) / max(1, steps1[-1]["step"] - steps1[0]["step"])
    first2 = next(e for e in run2 if "loss" in e)
    resumed = first2["resumed_from"]
    lost = fail["step"] - (resumed + 1 if resumed is not None else 0)
    return {"failed_at": fail["step"], "resumed_from": resumed, "rpo_steps": lost, "rpo_s": lost * sec_per_step,
            "rto_s": restart_s + first2["t"], "sec_per_step": sec_per_step}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run1")
    ap.add_argument("run2")
    ap.add_argument("--restart-s", type=float, default=0.0)
    a = ap.parse_args()
    r = analyze(events(a.run1), events(a.run2), a.restart_s)
    print("| failed at step | resumed from | RPO (steps) | RPO (s) | RTO (s) |\n|---|---|---|---|---|")
    print(f"| {r['failed_at']} | {r['resumed_from']} | {r['rpo_steps']} | {r['rpo_s']:.1f} | {r['rto_s']:.1f} |")
