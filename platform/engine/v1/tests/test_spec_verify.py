"""Speculative verification preserves the target distribution (P6.5 exercise 1, T0 NumPy version)."""
import numpy as np

from s2s_engine.spec_verify import verify_batch


def test_first_token_marginal_equals_target():
    rng = np.random.default_rng(0)
    V, k, B = 5, 3, 4000
    p = rng.dirichlet(np.ones(V), size=k + 1)
    q = rng.dirichlet(np.ones(V), size=k)
    draft = np.stack([[rng.choice(V, p=q[i]) for i in range(k)] for _ in range(B)])
    out = verify_batch(draft, np.broadcast_to(q, (B, k, V)), np.broadcast_to(p, (B, k + 1, V)), rng)
    first = np.bincount([o[0] for o in out], minlength=V) / B
    np.testing.assert_allclose(first, p[0], atol=0.03)
    assert all(1 <= len(o) <= k + 1 for o in out)


def test_identical_draft_accepts_everything():
    rng = np.random.default_rng(1)
    V, k = 6, 4
    p = rng.dirichlet(np.ones(V), size=k + 1)
    draft = np.array([[rng.choice(V, p=p[i]) for i in range(k)]])
    out = verify_batch(draft, p[None, :k], p[None], rng)
    assert out[0][:k] == list(draft[0]) and len(out[0]) == k + 1
