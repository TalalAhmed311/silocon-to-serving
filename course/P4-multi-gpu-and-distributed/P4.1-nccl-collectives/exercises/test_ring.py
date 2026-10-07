import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_p = HERE / ("solutions/ring.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "ring.py")
_spec = importlib.util.spec_from_file_location("ring_ut", _p)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def ranks(n, size, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(size) for _ in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("size", [8, 1000, 1001])
def test_all_reduce(n, size):
    bufs = ranks(n, size)
    want = np.sum(bufs, axis=0)
    out, steps = m.all_reduce([b.copy() for b in bufs])
    assert steps == 2 * (n - 1)
    for b in out:
        np.testing.assert_allclose(b, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [2, 5])
def test_reduce_scatter_owns_chunk_r(n):
    bufs = ranks(n, 103, seed=1)
    want = np.array_split(np.sum(bufs, axis=0), n)
    owned, steps = m.reduce_scatter([b.copy() for b in bufs])
    assert steps == n - 1
    for r in range(n):
        np.testing.assert_allclose(owned[r], want[r], rtol=1e-12, atol=1e-12)


def test_all_gather_order():
    chunks = [np.full(3, float(r)) for r in range(4)]
    out, steps = m.all_gather(chunks)
    assert steps == 3
    for b in out:
        assert b.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_rs_then_ag_equals_all_reduce():
    """Exercise 4's identity, in the simulator: all-reduce = reduce-scatter + all-gather (same bytes, same steps)."""
    bufs = ranks(6, 600, seed=2)
    owned, s1 = m.reduce_scatter([b.copy() for b in bufs])
    gathered, s2 = m.all_gather(owned)
    ar, s3 = m.all_reduce([b.copy() for b in bufs])
    assert s1 + s2 == s3
    for g, a in zip(gathered, ar):
        np.testing.assert_allclose(g, a, rtol=1e-12, atol=1e-12)
