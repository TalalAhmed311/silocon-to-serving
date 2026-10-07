"""01_metrics_from_log.py — per-request TTFT/TPOT/E2E and percentiles from a JSONL log of token timestamps.

Log format (one request per line): {"id": str, "arrival": seconds, "tokens": [time of each output token, ...]}
Run:      uv run python course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/examples/01_metrics_from_log.py \
            course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics/examples/sample_tokens.jsonl
Expected: deterministic tables computed from the file (r3 and r6 are the slow-TTFT tail).
Hardware: T0.
"""
import json
import sys

import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("| request | output tokens | TTFT ms | TPOT ms | E2E ms |\n|---|---|---|---|---|")
ttft, tpot, e2e, itl = [], [], [], []
for r in rows:
    t = r["tokens"]
    a = (t[0] - r["arrival"]) * 1e3
    e = (t[-1] - r["arrival"]) * 1e3
    p = (e - a) / (len(t) - 1) if len(t) > 1 else float("nan")   # undefined for a 1-token answer
    itl += [(y - x) * 1e3 for x, y in zip(t, t[1:])]
    ttft.append(a); e2e.append(e)
    if len(t) > 1:
        tpot.append(p)
    print(f"| {r['id']} | {len(t)} | {a:.0f} | {p:.1f} | {e:.0f} |")
print("\n| metric | n | mean | p50 | p90 | p99 |\n|---|---|---|---|---|---|")
for name, v in (("TTFT ms", ttft), ("TPOT ms", tpot), ("ITL ms (per token gap)", itl), ("E2E ms", e2e)):
    v = np.array(v)
    print(f"| {name} | {len(v)} | {v.mean():.1f} | {np.percentile(v, 50):.1f} | {np.percentile(v, 90):.1f} | {np.percentile(v, 99):.1f} |")
span = max(r["tokens"][-1] for r in rows) - min(r["arrival"] for r in rows)
print(f"\nthroughput: {len(rows) / span:.2f} req/s, {sum(len(r['tokens']) for r in rows) / span:.1f} output tok/s over {span:.2f} s")
print("Note how the mean TTFT sits far above the median: two slow requests dominate it.")
