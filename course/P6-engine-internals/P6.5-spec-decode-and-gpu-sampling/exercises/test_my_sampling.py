"""Tests for exercises/my_sampling.py (S2S_SOLUTIONS=1: against the reference). T0; the torch tests need [torch]."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform" / "engine" / "v1"))
from s2s_engine.sampling import filter_probs_np, top_p_threshold_sorted  # noqa: E402
from s2s_engine.sequence import SamplingParams  # noqa: E402

if os.environ.get("S2S_SOLUTIONS") == "1":
    from s2s_engine.sampling import top_p_threshold_np as top_p_threshold  # noqa: E402
    from s2s_engine.spec_verify import verify_batch  # noqa: E402
else:
    _spec = importlib.util.spec_from_file_location("my_sampling", Path(__file__).with_name("my_sampling.py"))
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    top_p_threshold, verify_batch = _m.top_p_threshold, _m.verify_batch


@pytest.mark.parametrize("seed", range(10))
def test_top_p_same_set_as_sort(seed):
    probs = np.random.default_rng(seed).dirichlet(np.full(500, 0.2))
    for p in (0.05, 0.5, 0.9, 0.999):
        assert ((probs >= top_p_threshold(probs, p)) == (probs >= top_p_threshold_sorted(probs, p))).all()


def test_top_p_peaked():
    probs = np.array([0.97, 0.01, 0.01, 0.01])
    assert (probs >= top_p_threshold(probs, 0.9)).sum() == 1


@pytest.mark.parametrize("seed", range(3))
def test_verify_preserves_first_token_distribution(seed):
    rng = np.random.default_rng(seed)
    V, k, B = 6, 3, 5000
    p = rng.dirichlet(np.ones(V), size=k + 1)
    q = rng.dirichlet(np.ones(V), size=k)
    draft = np.stack([[rng.choice(V, p=q[i]) for i in range(k)] for _ in range(B)])
    out = verify_batch(draft, np.broadcast_to(q, (B, k, V)), np.broadcast_to(p, (B, k + 1, V)), rng)
    np.testing.assert_allclose(np.bincount([o[0] for o in out], minlength=V) / B, p[0], atol=0.03)


def test_verify_lengths_and_prefix():
    rng = np.random.default_rng(0)
    V, k, B = 5, 4, 200
    p = rng.dirichlet(np.ones(V), size=(B, k + 1))
    q = rng.dirichlet(np.ones(V), size=(B, k))
    draft = rng.integers(0, V, size=(B, k))
    for b, o in enumerate(verify_batch(draft, q, p, rng)):
        assert 1 <= len(o) <= k + 1
        assert o[:-1] == list(draft[b, :len(o) - 1])            # everything but the last token is accepted draft


@pytest.mark.torch
def test_torch_sampler_matches_numpy_filters():
    torch = pytest.importorskip("torch")
    from s2s_engine.sampling import sample_torch
    rng = np.random.default_rng(0)
    V, n = 12, 30000
    logits = rng.standard_normal(V).astype(np.float32) * 2
    sp = SamplingParams(temperature=0.8, top_k=8, top_p=0.85, min_p=0.05)
    want = filter_probs_np(logits, sp)
    B = 1000
    rows = torch.tensor(logits).repeat(B, 1)
    g = torch.Generator().manual_seed(0)
    full = lambda v, dt=torch.float32: torch.full((B,), v, dtype=dt)  # noqa: E731
    toks = torch.cat([sample_torch(rows, full(0.8), full(8, torch.long), full(0.85), full(0.05), g) for _ in range(n // B)])
    got = np.bincount(toks.numpy(), minlength=V) / len(toks)
    np.testing.assert_allclose(got, want, atol=0.012)
    assert got[want == 0].sum() == 0
