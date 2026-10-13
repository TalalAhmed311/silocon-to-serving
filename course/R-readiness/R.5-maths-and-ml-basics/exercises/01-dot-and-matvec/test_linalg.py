import numpy as np
import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "linalg")


def test_dot_by_hand():
    assert m.dot([1, 2, 3], [4, 5, 6]) == 32      # 4 + 10 + 18
    assert m.dot([1, 0], [0, 1]) == 0              # perpendicular
    assert m.dot([], []) == 0


def test_dot_length_mismatch():
    with pytest.raises(ValueError):
        m.dot([1, 2], [1, 2, 3])


def test_matvec_by_hand():
    W = [[1, 2, 3], [4, 5, 6]]
    assert m.matvec(W, [1, 0, -1]) == [-2, -2]


def test_matvec_matches_numpy():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((7, 5))
    x = rng.standard_normal(5)
    got = m.matvec(W.tolist(), x.tolist())
    np.testing.assert_allclose(got, W @ x, rtol=1e-9, atol=1e-12)


def test_flops():
    assert m.matvec_flops(2, 3) == 12
    # one q_proj of Llama-3-8B: 4096 x 4096
    assert m.matvec_flops(4096, 4096) == 33_554_432
