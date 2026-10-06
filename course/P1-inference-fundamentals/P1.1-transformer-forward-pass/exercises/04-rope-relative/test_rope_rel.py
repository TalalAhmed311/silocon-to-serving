import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "rope_rel")
rng = np.random.default_rng(0)


def test_shift_invariance():
    q, k = rng.standard_normal(64), rng.standard_normal(64)
    for _ in range(20):
        a, b, s = (int(x) for x in rng.integers(0, 2000, 3))
        assert np.isclose(m.score(q, k, a, b), m.score(q, k, a + s, b + s), rtol=1e-4, atol=1e-8)


def test_not_absolute_invariant():
    q, k = rng.standard_normal(64), rng.standard_normal(64)
    assert not np.isclose(m.score(q, k, 5, 9), m.score(q, k, 5, 50), rtol=1e-3)


def test_offset_form():
    q, k = rng.standard_normal(32), rng.standard_normal(32)
    for a, b in [(0, 0), (3, 10), (100, 7), (4096, 4100)]:
        assert np.isclose(m.score(q, k, a, b), m.score_via_offset(q, k, b - a), rtol=1e-6, atol=1e-9)
