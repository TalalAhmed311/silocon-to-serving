import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

HERE = Path(__file__).resolve().parent
path = HERE / "solutions/verify.py" if os.environ.get("S2S_SOLUTIONS") == "1" else HERE / "verify.py"
spec = importlib.util.spec_from_file_location("verify_ut", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
V, K = 5, 3
rng0 = np.random.default_rng(0)
P = rng0.dirichlet(np.ones(V) * 0.7, size=K + 1)
Q = rng0.dirichlet(np.ones(V) * 0.7, size=K)


def verify_buggy(p, q, proposals, rng):
    out = []
    for i, t in enumerate(proposals):
        if rng.random() < min(1.0, p[i][t] / q[i][t]):
            out.append(t)
            continue
        out.append(int(rng.choice(len(p[i]), p=p[i])))   # BUG
        return out
    out.append(int(rng.choice(p.shape[1], p=p[len(proposals)])))
    return out


def first_tokens(fn, n=20000, seed=1):
    rng, counts = np.random.default_rng(seed), np.zeros(V)
    for _ in range(n):
        props = [int(rng.choice(V, p=Q[i])) for i in range(K)]   # proposals drawn from the draft
        counts[fn(P, Q, props, rng)[0]] += 1
    return counts


def test_first_token_matches_target():
    c = first_tokens(m.verify)
    assert chisquare(c, P[0] * c.sum()).pvalue > 1e-3


def test_buggy_rule_is_detected():
    c = first_tokens(verify_buggy)
    assert chisquare(c, P[0] * c.sum()).pvalue < 1e-3


def test_identical_models_accept_everything():
    rng = np.random.default_rng(2)
    props = [0, 1, 2]
    out = m.verify(np.vstack([Q, P[-1:]]), Q, props, rng)
    assert out[:3] == props and len(out) == 4
