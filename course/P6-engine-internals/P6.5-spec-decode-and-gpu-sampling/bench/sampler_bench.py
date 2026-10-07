"""Sort-based vs threshold-bisection top-p sampling, batched (P6.5 exercise 3 starter). T0 on CPU, T2 on GPU.

Run: uv run --extra torch python course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/bench/sampler_bench.py --device cuda
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform" / "engine" / "v1"))
from s2s_engine.sampling import sample_torch  # noqa: E402


def sort_sampler(logits, temperature, top_p, generator=None):
    probs = torch.softmax(logits.float() / temperature[:, None], -1)
    s, idx = probs.sort(-1, descending=True)
    c = s.cumsum(-1)
    s = torch.where(c - s >= top_p[:, None], torch.zeros_like(s), s)       # drop tokens after the mass is reached
    choice = torch.multinomial(s / s.sum(-1, keepdim=True), 1, generator=generator)
    return idx.gather(-1, choice).squeeze(-1)


def timeit(fn, device, reps=20):
    for _ in range(3):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
print(f"device={a.device} | V | B | sort (ms) | bisection (ms)")
for V in (32_000, 128_000):
    for B in (1, 64, 256):
        x = torch.randn(B, V, device=a.device)
        T = torch.full((B,), 0.8, device=a.device)
        P = torch.full((B,), 0.9, device=a.device)
        K = torch.zeros(B, dtype=torch.long, device=a.device)
        M = torch.zeros(B, device=a.device)
        ts = timeit(lambda: sort_sampler(x, T, P), a.device)
        tb = timeit(lambda: sample_torch(x, T, K, P, M), a.device)
        print(f"{V:7d} | {B:3d} | {ts:9.2f} | {tb:9.2f}")
