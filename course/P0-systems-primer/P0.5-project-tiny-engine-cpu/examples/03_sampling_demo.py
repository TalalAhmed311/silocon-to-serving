"""03_sampling_demo.py — how temperature and top-p reshape a next-token distribution.

Run:      uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/examples/03_sampling_demo.py
Expected: a table: as T falls, the top token's probability rises toward 1; as top-p falls, fewer tokens survive.
          Exact values below are computed, not measured, so they are the same on every machine.
Hardware: T0.
"""
import numpy as np

logits = np.array([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0])


def probs(T: float) -> np.ndarray:
    z = logits / T
    e = np.exp(z - z.max())
    return e / e.sum()


def nucleus(p: np.ndarray, top_p: float) -> np.ndarray:
    order = np.argsort(-p, kind="stable")
    keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p) + 1)]
    q = np.zeros_like(p)
    q[keep] = p[keep] / p[keep].sum()
    return q


print("| T | top_p | p(top) | tokens kept | entropy (nats) |\n|---|---|---|---|---|")
for T in (0.25, 0.5, 1.0, 1.5):
    for top_p in (1.0, 0.9, 0.5):
        q = nucleus(probs(T), top_p)
        nz = q[q > 0]
        print(f"| {T} | {top_p} | {q.max():.3f} | {len(nz)} | {-(nz * np.log(nz)).sum():.3f} |")
