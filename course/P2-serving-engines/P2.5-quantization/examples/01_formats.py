"""01_formats.py — round-trip error of the same tensor in FP16, BF16, FP8 (E4M3/E5M2, emulated), INT8, INT4.

Run:      uv run python course/P2-serving-engines/P2.5-quantization/examples/01_formats.py
Data:     Gaussian weights (std 0.02) with 0.1% outliers at 10× — a typical LLM weight shape (seeded, deterministic).
Output:   | format | bits/weight | relative RMS error | max abs error |
Hardware: T0. FP8 uses the exercise-2 reference emulator; BF16 is emulated by truncating fp32 mantissas (round-to-nearest-even).
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fp8", HERE / "exercises/solutions/fp8.py")
fp8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp8)
spec = importlib.util.spec_from_file_location("q4", HERE / "exercises/solutions/int4.py")
q4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q4)

rng = np.random.default_rng(0)
w = (0.02 * rng.standard_normal((1024, 1024))).astype(np.float32)
idx = rng.choice(w.size, w.size // 1000, replace=False)
w.flat[idx] *= 10                                   # outliers


def bf16(x):
    b = x.astype(np.float32).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000  # round to nearest even on the 16 dropped bits
    return b.view(np.float32)


def int8_per_channel(x):
    s = np.abs(x).max(axis=1, keepdims=True) / 127
    return np.clip(np.round(x / s), -127, 127) * s


def fp8_per_channel(x, fmt):
    mx = 448.0 if fmt == "e4m3" else 57344.0
    s = np.abs(x).max(axis=1, keepdims=True) / mx    # per-channel scale so each row uses the full FP8 range
    return fp8.quantize(x / s, fmt) * s


rows = [("fp16", 16, w.astype(np.float16).astype(np.float32)), ("bf16", 16, bf16(w)),
        ("fp8 e4m3 (per-channel scale)", 8, fp8_per_channel(w, "e4m3")),
        ("fp8 e5m2 (per-channel scale)", 8, fp8_per_channel(w, "e5m2")),
        ("int8 per-channel", 8, int8_per_channel(w)),
        ("int4 per-tensor (symmetric)", 4, q4.dequantize(*q4.quantize(w.reshape(1, -1), group=w.size, symmetric=True)).reshape(w.shape)),
        ("int4 group 128 (symmetric)", 4.125, q4.dequantize(*q4.quantize(w, group=128, symmetric=True))),
        ("int4 group 128 (asymmetric)", 4.25, q4.dequantize(*q4.quantize(w, group=128, symmetric=False)))]
print("| format | bits/weight | relative RMS error | max abs error |\n|---|---|---|---|")
for name, bits, y in rows:
    rel = np.sqrt(np.mean((y - w) ** 2)) / np.sqrt(np.mean(w ** 2))
    print(f"| {name} | {bits} | {rel:.2e} | {np.abs(y - w).max():.2e} |")
print("\nScales add bits: one fp16 scale per 128 weights = +0.125 bits/weight (+ a zero-point for asymmetric).")
