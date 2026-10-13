import warnings

import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "softmax")


def test_two_equal_scores():
    np.testing.assert_allclose(m.softmax(np.array([0.0, 0.0])), [0.5, 0.5])


def test_known_values():
    # exp(0)=1, exp(1)=2.71828..., sum=3.71828...
    got = m.softmax(np.array([0.0, 1.0]))
    np.testing.assert_allclose(got, [0.2689414, 0.7310586], rtol=1e-6)


def test_large_logits_do_not_overflow():
    x = np.array([1000.0, 1001.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # an overflow warning fails the test
        got = m.softmax(x)
    np.testing.assert_allclose(got, [0.2689414, 0.7310586], rtol=1e-5)


def test_naive_version_breaks_on_large_logits():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = m.naive_softmax(np.array([1000.0, 1001.0], dtype=np.float32))
    assert np.isnan(bad).any()


def test_shift_invariance():
    x = np.array([0.3, -1.2, 2.0, 0.0])
    np.testing.assert_allclose(m.softmax(x), m.softmax(x + 500.0), rtol=1e-9)


def test_axis_and_sums():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 6, 5))
    for axis in (-1, 0, 1):
        p = m.softmax(x, axis=axis)
        assert p.shape == x.shape
        np.testing.assert_allclose(p.sum(axis=axis), 1.0, rtol=1e-9)
        assert (p > 0).all()
