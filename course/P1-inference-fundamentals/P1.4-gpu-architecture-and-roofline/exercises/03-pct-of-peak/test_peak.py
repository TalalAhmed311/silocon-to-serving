import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "peak")


def test_bandwidth_and_compute():
    assert m.pct_of_peak(150, "L4", "bandwidth") == pytest.approx(50.0, rel=0.5)  # spec ~300 GB/s
    assert m.pct_of_peak(60.5, "L4", "compute") == pytest.approx(50.0, rel=0.5)


def test_null_spec_returns_none():
    assert m.pct_of_peak(100, "A10G", "compute") is None
    assert m.pct_of_peak(100, "T4", "compute", precision="fp8") is None


def test_annotation_flags_unverified():
    s = m.annotate(150, "L4", "bandwidth")
    assert "%" in s and ("UNVERIFIED" in s or "peak" in s)
    assert m.annotate(1, "A10G", "compute") == "1.0 (no spec)"
