"""01_spec_math.py — expected tokens per target pass and speedup vs (α, k, c); best k per (α, c). Formula values only.

Run: uv run python course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/01_spec_math.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform"))
from specdec.core import expected_speedup, expected_tokens_per_round  # noqa: E402

print("| α | k | E[tokens/pass] | speedup c=0.05 | speedup c=0.2 |\n|---|---|---|---|---|")
for a, k in [(0.6, 4), (0.8, 4), (0.8, 8), (0.9, 4), (0.9, 8)]:
    print(f"| {a} | {k} | {expected_tokens_per_round(a, k):.2f} | {expected_speedup(a, k, 0.05):.2f}x | {expected_speedup(a, k, 0.2):.2f}x |")
print("\n| α | c | best k (1..16) | speedup |\n|---|---|---|---|")
for a in (0.5, 0.7, 0.9):
    for c in (0.02, 0.1, 0.3):
        best = max(range(1, 17), key=lambda k: expected_speedup(a, k, c))
        print(f"| {a} | {c} | {best} | {expected_speedup(a, best, c):.2f}x |")
