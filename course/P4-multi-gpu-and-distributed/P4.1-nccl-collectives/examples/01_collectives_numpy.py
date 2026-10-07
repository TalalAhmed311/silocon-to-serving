"""01_collectives_numpy.py — simulate ring all-reduce on N fake ranks and check it against sum (T0).

Run: uv run python course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/examples/01_collectives_numpy.py --ranks 4
Expected: "ring all-reduce OK: 4 ranks, 6 steps (2(N-1)), each rank sent 1.50× its buffer" — the step count and the
bytes-sent ratio 2(N-1)/N are the two numbers the cost model is built on.
"""
from __future__ import annotations

import argparse

import numpy as np


def ring_allreduce(bufs: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    """In-place ring all-reduce on a list of equal-length arrays. Returns (bufs, steps, elements sent per rank)."""
    n = len(bufs)
    chunks = [np.array_split(b, n) for b in bufs]          # views: rank r's buffer split into n chunks
    steps = sent = 0
    # reduce-scatter: after n-1 steps, rank r owns the full sum of chunk (r+1) % n
    for s in range(n - 1):
        msgs = [(r, (r - s) % n, chunks[r][(r - s) % n].copy()) for r in range(n)]   # everyone sends simultaneously
        for r, c, data in msgs:
            chunks[(r + 1) % n][c] += data
        steps += 1
        sent += len(msgs[0][2])
    # all-gather: circulate the reduced chunks for n-1 more steps
    for s in range(n - 1):
        msgs = [(r, (r + 1 - s) % n, chunks[r][(r + 1 - s) % n].copy()) for r in range(n)]
        for r, c, data in msgs:
            chunks[(r + 1) % n][c][:] = data
        steps += 1
        sent += len(msgs[0][2])
    return bufs, steps, sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--size", type=int, default=1 << 16)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    bufs = [rng.standard_normal(a.size) for _ in range(a.ranks)]
    want = np.sum(bufs, axis=0)
    out, steps, sent = ring_allreduce([b.copy() for b in bufs])
    for b in out:
        np.testing.assert_allclose(b, want, rtol=1e-10, atol=1e-10)
    print(f"ring all-reduce OK: {a.ranks} ranks, {steps} steps (2(N-1)), each rank sent {sent / a.size:.2f}× its buffer")


if __name__ == "__main__":
    main()
