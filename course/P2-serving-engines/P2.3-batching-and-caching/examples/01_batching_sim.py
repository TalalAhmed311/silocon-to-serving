"""01_batching_sim.py — static vs continuous batching (and chunked prefill) on one seeded arrival trace.

Run:      uv run python course/P2-serving-engines/P2.3-batching-and-caching/examples/01_batching_sim.py
Output:   a table of mean / p90 TTFT and E2E per policy, the worst ITL, and results/batching.png (batch-size timeline).
Hardware: T0. Uses the reference simulator (exercises/solutions/batchsim.py) — exercise 1 asks you to write it.
"""
import importlib.util
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bs", HERE / "exercises/solutions/batchsim.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

rng = random.Random(0)
t, reqs = 0.0, []
for i in range(200):
    t += rng.expovariate(1 / 40)                         # one arrival per 40 ms on average (25 req/s)
    reqs.append(bs.Req(i, t, prompt=rng.choice([64, 256, 1024, 4000]), output=rng.choice([8, 32, 128, 512])))

runs = {"static B=16": bs.static(reqs, 16), "continuous max=16": bs.continuous(reqs, 16),
        "continuous + chunk 512": bs.continuous(reqs, 16, chunk_tokens=512)}
print("| policy | mean TTFT ms | p90 TTFT ms | mean E2E ms | p90 E2E ms | worst ITL ms |\n|---|---|---|---|---|---|")
for name, o in runs.items():
    tt, ee = np.array(list(o.ttft(reqs).values())), np.array(list(o.e2e(reqs).values()))
    worst = max(o.max_itl(r.rid) for r in reqs)
    print(f"| {name} | {tt.mean():.0f} | {np.percentile(tt, 90):.0f} | {ee.mean():.0f} | {np.percentile(ee, 90):.0f} | {worst:.0f} |")

fig, ax = plt.subplots(len(runs), 1, figsize=(10, 7), sharex=True)
for a, (name, o) in zip(ax, runs.items()):
    a.step([s[0] for s in o.steps], [s[2] for s in o.steps], where="post")
    a.set_ylabel("decoding")
    a.set_title(name, fontsize=9)
ax[-1].set_xlabel("time (ms)")
Path("results").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("results/batching.png", dpi=120)
print("wrote results/batching.png — note the idle slots in static batching after short requests finish")
