"""check_scale_log.py — verify a scale-up/scale-down timeline from `kubectl get pods -w --output-watch-events`.

Input: a log where each line is prefixed with a unix timestamp (pipe through `ts '%.s'` from moreutils), e.g.
    1760000000.12 ADDED  vllm-6d…-abcde  0/1  Pending  0  0s  <none>  <none>
Usage: python check_scale_log.py results/p3.6-watch.log --deployment vllm [--max-replicas 4] [--expect-zero]
Prints the replica-count timeline (Ready pods) and checks: it scaled up, never exceeded max, and (with
--expect-zero) returned to zero. Exit code 1 on failure, for use as the exercise 2/3 test.
"""
from __future__ import annotations

import argparse
import re
import sys

LINE = re.compile(r"^(?P<t>\d+(\.\d+)?)\s+(?P<ev>ADDED|MODIFIED|DELETED)\s+(?:pod/)?(?P<pod>\S+)\s+(?P<ready>\d+)/(?P<total>\d+)\s+(?P<status>\S+)")


def timeline(lines, deployment):
    ready, out = {}, []
    for ln in lines:
        m = LINE.match(ln.strip())
        if not m or not m["pod"].startswith(deployment + "-"):
            continue
        t, pod = float(m["t"]), m["pod"]
        is_ready = m["ev"] != "DELETED" and m["status"] == "Running" and m["ready"] == m["total"]
        ready[pod] = is_ready
        if m["ev"] == "DELETED":
            ready.pop(pod, None)
        n = sum(ready.values())
        if not out or out[-1][1] != n:
            out.append((t, n))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--deployment", default="vllm")
    ap.add_argument("--max-replicas", type=int, default=4)
    ap.add_argument("--expect-zero", action="store_true")
    a = ap.parse_args(argv)
    tl = timeline(open(a.log).read().splitlines(), a.deployment)
    if not tl:
        print("FAIL: no matching pod lines")
        return 1
    t0 = tl[0][0]
    print("| t (s) | ready replicas |\n|---|---|")
    for t, n in tl:
        print(f"| {t - t0:.0f} | {n} |")
    peak = max(n for _, n in tl)
    errs = []
    if peak < 2:
        errs.append("never scaled above 1 replica")
    if peak > a.max_replicas:
        errs.append(f"exceeded max ({peak} > {a.max_replicas})")
    if a.expect_zero and tl[-1][1] != 0:
        errs.append(f"did not return to zero (last = {tl[-1][1]})")
    for e in errs:
        print("FAIL:", e)
    print("OK" if not errs else "")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
