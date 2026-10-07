"""01_tp_mlp_numpy.py — split a SwiGLU MLP across simulated TP ranks and check it against the unsplit MLP (T0).

Run: uv run python course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/01_tp_mlp_numpy.py
Expected: for tp = 1, 2, 4, 8 the max |error| vs unsplit is ~1e-6 (fp32), each rank holds 1/tp of the MLP weights, and
the only communication is ONE all-reduce of a [tokens, d] tensor. The "wrong order" split shows a large error.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel as P  # noqa: E402

d, f, T = 512, 1536, 8
rng = np.random.default_rng(0)
wg, wu = (rng.standard_normal((d, f)).astype(np.float32) / np.sqrt(d) for _ in range(2))
wd = rng.standard_normal((f, d)).astype(np.float32) / np.sqrt(f)
x = rng.standard_normal((T, d)).astype(np.float32)
ref = P.mlp(x, wg, wu, wd)
print("| tp | weights per rank (MB) | all-reduce bytes | max abs err |\n|---|---|---|---|")
for tp in (1, 2, 4, 8):
    _, out = P.tp_mlp(x, wg, wu, wd, tp)
    print(f"| {tp} | {3 * d * f * 4 / tp / 1e6:.2f} | {T * d * 4 if tp > 1 else 0} | {np.abs(out - ref).max():.2e} |")
print(f"wrong split order (row-split first): max abs err {np.abs(P.tp_mlp_wrong_order(x, wg, wu, wd, 4) - ref).max():.2e}")
