"""plot.py — latency vs throughput curve from results/loadgen.json (the knee plot for #4 and #15).

Run: PYTHONPATH=platform uv run python -m loadgen.plot results/loadgen.json [more.json ...]   -> results/knee.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for f in sys.argv[1:]:
    d = json.loads(Path(f).read_text())
    rows, label = d["rows"], Path(f).stem
    ax[0].plot([r["out_tok_s"] for r in rows], [r["ttft_p90"] for r in rows], marker="o", label=label)
    ax[1].plot([r["offered_rps"] for r in rows], [r["goodput_rps"] for r in rows], marker="o", label=label)
    if d.get("knee"):
        ax[0].axvline(d["knee"]["out_tok_s"], ls=":", c="gray")
ax[0].set(xlabel="output tokens/s", ylabel="p90 TTFT (s)", title="latency vs throughput")
ax[1].plot([0, max(r["offered_rps"] for r in rows)], [0, max(r["offered_rps"] for r in rows)], ls="--", c="gray", label="ideal")
ax[1].set(xlabel="offered req/s", ylabel="goodput req/s", title="goodput")
for a in ax:
    a.grid(alpha=0.3)
    a.legend()
Path("results").mkdir(exist_ok=True)
plt.savefig("results/knee.png", dpi=130, bbox_inches="tight")
print("wrote results/knee.png")
