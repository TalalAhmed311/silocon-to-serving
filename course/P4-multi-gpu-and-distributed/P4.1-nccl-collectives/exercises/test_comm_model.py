"""Exercises 2 and 3: the α–β fit and the busbw factors (T0, synthetic inputs — no hardware numbers here)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOD))
import commmodel as cm  # noqa: E402

_spec = importlib.util.spec_from_file_location("parse_nccl", MOD / "bench/parse_nccl_tests.py")
pn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pn)


def test_busbw_factors():
    n = 8
    assert cm.busbw("all_reduce", 10.0, n) == pytest.approx(10.0 * 2 * 7 / 8)
    assert cm.busbw("all_gather", 10.0, n) == pytest.approx(10.0 * 7 / 8)
    assert cm.busbw("reduce_scatter", 10.0, n) == pytest.approx(10.0 * 7 / 8)
    assert cm.busbw("broadcast", 10.0, n) == 10.0


def test_ring_time_equals_rs_plus_ag():
    S, n, B, a = 64e6, 4, 100e9, 5e-6
    assert cm.ring_allreduce_time(S, n, B, a) == pytest.approx(2 * cm.ring_rs_or_ag_time(S, n, B, a))


def test_busbw_of_modelled_ring_is_link_bandwidth():
    """With α = 0, a ring all-reduce's busbw equals the link bandwidth — the reason busbw exists."""
    S, n, B = 1e9, 8, 50e9
    t = cm.ring_allreduce_time(S, n, B)
    assert cm.busbw("all_reduce", S / t, n) == pytest.approx(B)


def test_fit_recovers_alpha_beta():
    rng = np.random.default_rng(0)
    n, B, alpha = 4, 40e9, 8e-6
    sizes = [2 ** k for k in range(10, 31)]
    times = [cm.ring_allreduce_time(s, n, B, alpha) * (1 + rng.normal(0, 0.01)) for s in sizes]
    small = [(s, t) for s, t in zip(sizes, times) if s <= 64 << 10]      # latency regime → α
    large = [(s, t) for s, t in zip(sizes, times) if s >= 1 << 20]       # bandwidth regime → B
    a, _ = cm.fit_alpha_beta(*zip(*small))
    _, eff_bw = cm.fit_alpha_beta(*zip(*large))
    assert a / (2 * (n - 1)) == pytest.approx(alpha, rel=0.2)
    assert eff_bw * 2 * (n - 1) / n == pytest.approx(B, rel=0.05)


def test_tree_wins_small_ring_wins_large():
    n, B, alpha = 64, 25e9, 10e-6
    x = cm.crossover_size(n, B, alpha)
    assert cm.tree_allreduce_time(x / 10, n, B, alpha) < cm.ring_allreduce_time(x / 10, n, B, alpha)
    assert cm.tree_allreduce_time(x * 10, n, B, alpha) > cm.ring_allreduce_time(x * 10, n, B, alpha)


SYNTHETIC = """\
# nccl-tests style output — SYNTHETIC numbers for the parser test, not a measurement
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    20.00    0.00    0.00      0    19.00    0.00    0.00      0
     1048576        262144     float     sum      -1   100.00   10.49   15.73      0    99.00   10.59   15.89      0
# Avg bus bandwidth    : 7.86
"""


def test_parse_nccl_tests():
    rows = pn.parse(SYNTHETIC)
    assert [r["size"] for r in rows] == [8, 1048576]
    assert rows[1] == {"size": 1048576, "time_us": 100.0, "algbw": 10.49, "busbw": 15.73}
    assert "| 1048576 | 100.0 | 10.49 | 15.73 | 52% |" in pn.table(rows, 30.0)
