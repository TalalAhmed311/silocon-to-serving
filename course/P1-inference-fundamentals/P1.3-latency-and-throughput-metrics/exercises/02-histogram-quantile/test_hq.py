import math

import numpy as np

from s2s.exercise import load_impl

m = load_impl(__file__, "hq")
LES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def width_at(x):
    prev = 0.0
    for le in LES:
        if x <= le:
            return le - prev
        prev = le
    return math.inf


def test_close_to_exact_within_bucket_width():
    v = np.random.default_rng(0).lognormal(mean=-2.0, sigma=1.0, size=10_000)
    b = m.observe_into_buckets(v, LES)
    for q in (0.5, 0.9, 0.99):
        exact = float(np.percentile(v, 100 * q))
        assert abs(m.histogram_quantile(q, b) - exact) <= width_at(exact)


def test_edges():
    assert math.isnan(m.histogram_quantile(0.5, m.observe_into_buckets([], LES)))
    b = m.observe_into_buckets([100.0, 200.0], LES)            # all in +Inf
    assert m.histogram_quantile(0.9, b) == 10.0
    b = m.observe_into_buckets([0.003] * 10, LES)
    assert 0 <= m.histogram_quantile(0.0, b) <= 0.005
    assert m.histogram_quantile(1.0, b) == 0.005
