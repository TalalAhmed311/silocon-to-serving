"""Exercise 1 solution."""
import numpy as np


def verify(p, q, proposals, rng):
    out = []
    for i, t in enumerate(proposals):
        if rng.random() < min(1.0, p[i][t] / q[i][t]):
            out.append(t)
            continue
        r = np.maximum(p[i] - q[i], 0.0)
        out.append(int(rng.choice(len(r), p=r / r.sum())))
        return out
    out.append(int(rng.choice(p.shape[1], p=p[len(proposals)])))
    return out
