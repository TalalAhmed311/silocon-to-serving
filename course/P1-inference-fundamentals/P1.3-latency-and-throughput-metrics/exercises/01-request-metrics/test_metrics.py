import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "metrics")


def test_r1():
    r = m.request_metrics(0.0, [0.210, 0.240, 0.270, 0.300, 0.330])
    assert r["ttft"] == pytest.approx(0.210)
    assert r["e2e"] == pytest.approx(0.330)
    assert r["tpot"] == pytest.approx(0.030)
    assert r["itl"] == pytest.approx([0.03] * 4)
    assert r["n"] == 5


def test_single_token():
    r = m.request_metrics(0.4, [0.62])
    assert r["ttft"] == pytest.approx(0.22) and r["tpot"] is None and r["itl"] == []


def test_percentiles():
    p = m.percentiles([1, 2, 3, 4, 100])
    assert p[50] == pytest.approx(3.0)
    assert p[90] == pytest.approx(61.6)
