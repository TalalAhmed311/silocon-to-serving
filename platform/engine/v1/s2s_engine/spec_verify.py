"""spec_verify.py — batched speculative-decoding verification (P6.5): exact rejection sampling, NumPy reference.
Same rule as platform/specdec/core.py (P2.6) and the L6 #87 CUDA kernel; this version is vectorised over a batch and
is what the engine calls. Returns, per sequence, the accepted draft prefix plus one corrected/bonus token."""
from __future__ import annotations

import numpy as np


def verify_batch(draft: np.ndarray, q: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> list[list[int]]:
    """draft [B, k] ints; q [B, k, V] draft probs; p [B, k+1, V] target probs. → list of emitted token lists."""
    B, k = draft.shape
    out = []
    for b in range(B):
        toks = []
        for i in range(k):
            x = int(draft[b, i])
            if rng.random() < min(1.0, p[b, i, x] / max(q[b, i, x], 1e-30)):
                toks.append(x)
                continue
            resid = np.maximum(p[b, i] - q[b, i], 0.0)
            s = resid.sum()
            toks.append(int(rng.choice(resid.size, p=resid / s)) if s > 0 else int(np.argmax(p[b, i])))
            break
        else:
            toks.append(int(rng.choice(p.shape[-1], p=p[b, k])))   # all accepted: bonus token from the target
        out.append(toks)
    return out
