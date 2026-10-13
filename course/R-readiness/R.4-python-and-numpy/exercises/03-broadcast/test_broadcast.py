import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "broadcast")


def test_normalize_rows_matches_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(5, 8))
    out = m.normalize_rows(x)
    ref = np.stack([(r - r.mean()) / np.sqrt(r.var() + 1e-5) for r in x])
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
    assert out.shape == x.shape


def test_rows_have_zero_mean_and_unit_spread():
    rng = np.random.default_rng(1)
    out = m.normalize_rows(rng.normal(10.0, 5.0, size=(4, 64)), eps=0.0)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=1), 1.0, rtol=1e-12)


def test_known_values():
    out = m.normalize_rows(np.array([[1.0, 3.0]]), eps=0.0)       # mean 2, std 1
    np.testing.assert_allclose(out, [[-1.0, 1.0]])


def test_constant_row_stays_finite():
    out = m.normalize_rows(np.full((2, 4), 7.0))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 0.0)


def test_does_not_modify_input():
    x = np.array([[1.0, 2.0, 6.0]])
    before = x.copy()
    m.normalize_rows(x)
    assert np.array_equal(x, before)


def test_center_columns():
    x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    out = m.center_columns(x)
    np.testing.assert_allclose(out, [[-2.0, -10.0], [0.0, 0.0], [2.0, 10.0]])
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
