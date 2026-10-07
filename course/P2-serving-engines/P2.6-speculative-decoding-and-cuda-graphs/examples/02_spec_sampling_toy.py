"""02_spec_sampling_toy.py — speculative sampling is exact: compare next-token frequencies to the target's distribution.

Run: uv run python course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/02_spec_sampling_toy.py
Uses two random Markov "models" over a 6-token vocabulary (seeded). Prints χ² p-values for the correct rule and for
a deliberately broken rule (resample from p instead of max(0, p − q)).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform"))
from specdec import core  # noqa: E402

rng = np.random.default_rng(0)
V = 6
target, draft = core.Markov(rng.random((V, V)) ** 3), core.Markov(rng.random((V, V)) ** 3)
want = target.probs([2])


def first_token_counts(step_fn, n=20000):
    counts = np.zeros(V)
    r = np.random.default_rng(1)
    for _ in range(n):
        counts[step_fn([2], r)[0]] += 1
    return counts


correct = lambda pre, r: core.speculative_step(pre, draft, target, 3, r, core.Stats())  # noqa: E731


def broken(pre, r):
    q = draft.probs(pre)
    t = int(r.choice(V, p=q))
    p = target.probs(pre)
    if r.random() < min(1, p[t] / q[t]):
        return [t]
    return [int(r.choice(V, p=p))]          # BUG: should sample from norm(max(0, p − q))


for name, fn in (("correct rule", correct), ("broken rule", broken)):
    c = first_token_counts(fn)
    print(f"{name:13s}: χ² p-value = {chisquare(c, want * c.sum()).pvalue:.3g}   (p > 0.01 means consistent with the target)")
