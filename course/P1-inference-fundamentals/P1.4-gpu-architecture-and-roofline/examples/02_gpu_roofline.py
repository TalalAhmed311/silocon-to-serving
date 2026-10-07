"""02_gpu_roofline.py — rooflines (dense fp16 tensor) for every GPU in gpu_specs.yaml, with decode/prefill/attention placed.

Run:      uv run python course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/examples/02_gpu_roofline.py
Output:   results/gpu_roofline.png
Hardware: T0.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import specs  # noqa: E402

ops = {"decode B=1": 1, "decode B=64": 64, "attention (decode)": 1, "prefill T=2048": 2048}
x = np.logspace(-1, 4, 400)
plt.figure(figsize=(8, 5))
for g in specs.load():
    if g["fp16_dense_tflops"] is None:
        continue
    P, B = g["fp16_dense_tflops"] * 1e12, g["hbm_gbs"] * 1e9
    plt.loglog(x, np.minimum(P, x * B) / 1e12, label=f"{g['name']} ({g['status'].split()[0]})")
for name, i in ops.items():
    plt.axvline(i, ls=":", c="gray")
    plt.text(i * 1.05, 0.15, name, rotation=90, fontsize=8, color="gray")
plt.xlabel("arithmetic intensity (FLOP / byte)")
plt.ylabel("attainable TFLOP/s (dense fp16)")
plt.title("GPU rooflines from gpu_specs.yaml")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
Path("results").mkdir(exist_ok=True)
plt.savefig("results/gpu_roofline.png", dpi=130, bbox_inches="tight")
print("wrote results/gpu_roofline.png")
