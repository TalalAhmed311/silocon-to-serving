"""Reference ring collectives (P4.1 exercise 1)."""
from __future__ import annotations

import numpy as np


def reduce_scatter(bufs):
    n = len(bufs)
    chunks = [[c.copy() for c in np.array_split(b, n)] for b in bufs]
    # step s: rank r sends chunk (r - s - 1) mod n; after n-1 steps rank r owns the full sum of chunk r
    for s in range(n - 1):
        msgs = [(r, (r - s - 1) % n, chunks[r][(r - s - 1) % n].copy()) for r in range(n)]
        for r, c, data in msgs:
            chunks[(r + 1) % n][c] += data
    return [chunks[r][r] for r in range(n)], n - 1


def all_gather(chunks):
    n = len(chunks)
    have = [[None] * n for _ in range(n)]
    for r in range(n):
        have[r][r] = chunks[r].copy()
    # step s: rank r forwards chunk (r - s) mod n, the one it received last step (its own at s = 0)
    for s in range(n - 1):
        msgs = [(r, (r - s) % n, have[r][(r - s) % n]) for r in range(n)]
        for r, c, data in msgs:
            have[(r + 1) % n][c] = data.copy()
    return [np.concatenate(h) for h in have], n - 1


def all_reduce(bufs):
    owned, s1 = reduce_scatter(bufs)
    out, s2 = all_gather(owned)
    return out, s1 + s2
