import numpy as np
import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "distances")


@pytest.mark.parametrize("n,k,d", [(1, 1, 1), (3, 5, 2), (20, 7, 16), (50, 40, 3)])
def test_matches_loop(n, k, d):
    rng = np.random.default_rng(n * 100 + k)
    a, b = rng.normal(size=(n, d)), rng.normal(size=(k, d))
    got = m.pairwise_distances(a, b)
    assert got.shape == (n, k)
    np.testing.assert_allclose(got, m.pairwise_distances_loop(a, b), rtol=1e-6)


def test_known_values():
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0]])
    np.testing.assert_allclose(m.pairwise_distances(a, b), [[0.0], [5.0]])


def test_distance_to_self_is_zero():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(6, 4))
    np.testing.assert_allclose(np.diag(m.pairwise_distances(a, a)), 0.0, atol=1e-12)


def test_large_input_finishes():
    # 4,000,000 pairs: fine for a broadcast, painfully slow for a Python double loop.
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2000, 2)), rng.normal(size=(2000, 2))
    got = m.pairwise_distances(a, b)
    assert got.shape == (2000, 2000)
    np.testing.assert_allclose(got[5, 9], np.linalg.norm(a[5] - b[9]), rtol=1e-6)
