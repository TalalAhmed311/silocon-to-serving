"""Sampling filters and reproducibility (P6.5 exercise 2, NumPy side). T0. The torch version is tested on GPU."""
import numpy as np
import pytest

from s2s_engine.sampling import filter_probs_np, sample_np, top_p_threshold_np, top_p_threshold_sorted
from s2s_engine.sequence import SamplingParams


@pytest.mark.parametrize("seed", range(20))
def test_top_p_threshold_matches_sort(seed):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(200, 0.3))
    for p in (0.1, 0.5, 0.9, 0.99):
        keep_bisect = probs >= top_p_threshold_np(probs, p)
        keep_sort = probs >= top_p_threshold_sorted(probs, p)
        assert (keep_bisect == keep_sort).all()
        assert probs[keep_bisect].sum() >= p - 1e-9


def test_filters():
    logits = np.log(np.array([0.5, 0.2, 0.15, 0.1, 0.05]))
    assert (filter_probs_np(logits, SamplingParams(temperature=1, top_k=2)) > 0).sum() == 2
    assert (filter_probs_np(logits, SamplingParams(temperature=1, min_p=0.25)) > 0).sum() == 3   # ≥ 0.125
    assert (filter_probs_np(logits, SamplingParams(temperature=1, top_p=0.65)) > 0).sum() == 2
    pr = filter_probs_np(logits, SamplingParams(temperature=1))
    np.testing.assert_allclose(pr, np.exp(logits))


def test_greedy_and_seeded():
    logits = np.array([0.1, 3.0, 0.2])
    assert sample_np(logits, SamplingParams(temperature=0), np.random.default_rng(0)) == 1
    p = SamplingParams(temperature=1.0)
    a = [sample_np(logits, p, np.random.default_rng(5)) for _ in range(3)]
    b = [sample_np(logits, p, np.random.default_rng(5)) for _ in range(3)]
    assert a == b


def test_empirical_distribution():
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    rng = np.random.default_rng(0)
    n = 20000
    counts = np.bincount([sample_np(logits, SamplingParams(temperature=1), rng) for _ in range(n)], minlength=3)
    np.testing.assert_allclose(counts / n, [0.6, 0.3, 0.1], atol=0.015)
