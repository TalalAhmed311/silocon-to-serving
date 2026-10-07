"""03_arith_intensity_vs_batch.py — decode arithmetic intensity vs batch size at several context lengths, vs a GPU ridge.

Run:      uv run python course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/examples/03_arith_intensity_vs_batch.py --gpu L4
Output:   results/intensity_vs_batch.png + a table of the batch where decode first crosses the ridge (if ever).
Model:    Llama-3-8B-shaped preset (UNVERIFIED config values, see P1.1), fp16 weights and KV.
GPU:      peak and bandwidth from P1.4/gpu_specs.yaml (status: UNVERIFIED until checked against the whitepaper).
Hardware: T0.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SPECS = yaml.safe_load((ROOT / "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml").read_text())

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="L4", choices=[g["name"] for g in SPECS["gpus"]])
a = ap.parse_args()
g = next(x for x in SPECS["gpus"] if x["name"] == a.gpu)
ridge = g["fp16_dense_tflops"] * 1e12 / (g["hbm_gbs"] * 1e9)

P, d, L, kv_tok = 8.03e9, 4096, 32, 2 * 32 * 8 * 128 * 2  # params, hidden, layers, KV bytes/token (fp16)


def intensity(B: int, t: int) -> float:
    flops = 2 * P * B + B * 4 * d * L * t
    bytes_ = P * 2 + B * t * kv_tok
    return flops / bytes_


batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
plt.figure(figsize=(7, 4.5))
print(f"{a.gpu}: ridge = {ridge:.0f} FLOP/byte ({g['status']})\n\n| context t | first batch past the ridge |\n|---|---|")
for t in (256, 2048, 8192, 32768):
    ys = [intensity(B, t) for B in batches]
    plt.loglog(batches, ys, marker="o", label=f"t = {t}")
    cross = next((B for B, y in zip(batches, ys) if y >= ridge), None)
    print(f"| {t} | {cross if cross else 'never (≤ 512)'} |")
plt.axhline(ridge, ls="--", c="gray", label=f"{a.gpu} ridge")
plt.xlabel("decode batch size B")
plt.ylabel("arithmetic intensity (FLOP/byte)")
plt.title("Decode intensity: batching helps until the KV cache dominates")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
Path("results").mkdir(exist_ok=True)
plt.savefig("results/intensity_vs_batch.png", dpi=130, bbox_inches="tight")
print("\nwrote results/intensity_vs_batch.png")
