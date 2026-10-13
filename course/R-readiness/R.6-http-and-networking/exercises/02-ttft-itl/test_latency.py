import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "latency")


def test_even_stream():
    # sent at t=10.0, first token at 10.5, then one every 0.05 s
    arr = [10.5, 10.55, 10.60, 10.65, 10.70]
    r = m.stream_metrics(10.0, arr)
    assert r["ttft"] == pytest.approx(0.5)
    assert r["itl_mean"] == pytest.approx(0.05)
    assert r["itl_p90"] == pytest.approx(0.05)
    assert r["e2e"] == pytest.approx(0.7)


def test_one_slow_gap_shows_in_p90_more_than_mean():
    # gaps: 0.1 x9, then 1.0  -> mean 0.19; linear p90 at position 8.1 of 10 sorted gaps = 0.1 + 0.1*(1.0-0.1) = 0.19
    arr = [1.0]
    for g in [0.1] * 9 + [1.0]:
        arr.append(arr[-1] + g)
    r = m.stream_metrics(0.0, arr)
    assert r["itl_mean"] == pytest.approx(0.19)
    assert r["itl_p90"] == pytest.approx(0.19)


def test_p90_hand_value():
    # gaps 1,2,3,4,5 -> linear p90 position 0.9*4 = 3.6 -> 4 + 0.6*(5-4) = 4.6
    arr = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]
    assert m.stream_metrics(0.0, arr)["itl_p90"] == pytest.approx(4.6)


def test_single_chunk_has_no_gaps():
    r = m.stream_metrics(2.0, [2.4])
    assert r["ttft"] == pytest.approx(0.4)
    assert r["itl_mean"] is None and r["itl_p90"] is None
    assert r["e2e"] == pytest.approx(0.4)


def test_empty_raises():
    with pytest.raises(ValueError):
        m.stream_metrics(0.0, [])
