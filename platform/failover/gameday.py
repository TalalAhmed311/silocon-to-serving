"""gameday.py — the measured timeline of a failover drill (C1 exercise 3).

Input: a JSONL client log, one line per request: {"t": start (unix s), "end": end, "ok": bool, "region": serving
region or null} — the client is a #4-style loop running OUTSIDE both regions — plus the time you killed the region.
Output: detection time (kill → controller switch, if given), time to first success in
the other region, failover time (kill → steady success in the new region), requests failed, error-budget burn.

    python -m failover.gameday results/c1/client.jsonl --kill-t 1767225600 [--switch-t ...]
"""
from __future__ import annotations

import argparse
import json


def load(path):
    return [json.loads(l) for l in open(path) if l.strip().startswith("{")]


def analyze(reqs: list[dict], kill_t: float, switch_t: float | None = None, steady_n: int = 20) -> dict:
    reqs = sorted(reqs, key=lambda r: r["t"])
    before = [r for r in reqs if r["t"] < kill_t]
    old = max(set(r["region"] for r in before if r["ok"]), key=lambda g: sum(1 for r in before if r["region"] == g))
    after = [r for r in reqs if r["t"] >= kill_t]
    failed = [r for r in after if not r["ok"]]
    first_fail = failed[0]["t"] if failed else None
    new_ok = [r for r in after if r["ok"] and r["region"] != old]
    first_new = new_ok[0]["end"] if new_ok else None
    steady = None                                          # first time steady_n consecutive successes in the new region
    run = 0
    for r in after:
        if r["ok"] and r["region"] != old:
            run += 1
            if run == steady_n:
                steady = r["end"]
                break
        else:
            run = 0
    return {
        "old_region": old,
        "requests_after_kill": len(after),
        "failed": len(failed),
        "first_failure_s": None if first_fail is None else first_fail - kill_t,
        "detection_s": None if switch_t is None else switch_t - kill_t,
        "first_success_new_region_s": None if first_new is None else first_new - kill_t,
        "failover_s": None if steady is None else steady - kill_t,
        "error_rate_during_drill": len(failed) / max(1, len(after)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--kill-t", type=float, required=True)
    ap.add_argument("--switch-t", type=float)
    a = ap.parse_args()
    r = analyze(load(a.log), a.kill_t, a.switch_t)
    print("| metric | value |\n|---|---|")
    for k, v in r.items():
        print(f"| {k} | {v if not isinstance(v, float) else f'{v:.2f}'} |")
