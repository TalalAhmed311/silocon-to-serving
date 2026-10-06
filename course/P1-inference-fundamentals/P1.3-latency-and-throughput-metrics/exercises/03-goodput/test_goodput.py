import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "goodput")
R = [
    {"ttft": 0.2, "tpot": 0.03},
    {"ttft": 0.9, "tpot": 0.03},
    {"ttft": 0.3, "tpot": 0.20},   # fast start, slow stream: fails TPOT
    {"ttft": 0.4, "tpot": None},   # one-token answer
    {"ttft": 2.5, "tpot": 0.02},   # queued too long
]


def test_goodput():
    assert m.goodput(R, ttft_slo=1.0, tpot_slo=0.05, window_s=2.0) == pytest.approx(1.5)


def test_attainment():
    assert m.slo_attainment(R, 1.0, 0.05) == pytest.approx(0.6)
    assert m.slo_attainment([], 1.0, 0.05) == 0.0
