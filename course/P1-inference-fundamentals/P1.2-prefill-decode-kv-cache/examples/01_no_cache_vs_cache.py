"""01_no_cache_vs_cache.py — decode with a KV cache vs recomputing the whole prefix every step; same logits.

Run:      uv run python course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/examples/01_no_cache_vs_cache.py
Expected: "cached == uncached" for every position (max |Δ| < 1e-4) and a count of matvecs done by each.
Hardware: T0.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
V0 = ROOT / "platform/engine/v0"
spec = importlib.util.spec_from_file_location("llama_numpy", V0 / "reference/llama_numpy.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def uncached_logits(model_dir, tokens: list[int]) -> np.ndarray:
    """No cache: for the last token, rebuild K/V of the whole prefix from scratch (a fresh model = empty cache)."""
    m = ref.LlamaNumpy(model_dir)
    out = None
    for p, t in enumerate(tokens):      # every step re-processes the entire prefix
        out = m.forward(t, p)
    return out


def main() -> None:
    d = tempfile.mkdtemp()
    subprocess.run([sys.executable, str(V0 / "tools/make_tiny_llama.py"), d], check=True, capture_output=True)
    toks = [7, 3, 99, 12, 5, 64, 31, 2, 200, 17]
    cached = ref.LlamaNumpy(d)
    worst = 0.0
    for p, t in enumerate(toks):
        a = cached.forward(t, p)
        b = uncached_logits(d, toks[: p + 1])
        worst = max(worst, float(np.abs(a - b).max()))
    n = len(toks)
    print(f"cached == uncached at all {n} positions: max |Δ| = {worst:.2e}")
    assert worst < 1e-4
    print(f"forward passes: cached {n}, uncached {n * (n + 1) // 2} (= 1 + 2 + ... + {n}: O(n²))")


if __name__ == "__main__":
    main()
