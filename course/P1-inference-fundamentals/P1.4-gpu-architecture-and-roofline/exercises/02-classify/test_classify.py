import sys
from pathlib import Path

import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "classify")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import specs  # noqa: E402


def test_fixture_gpu():
    assert m.ridge(100, 1000) == pytest.approx(100)
    assert m.attainable_tflops(10, 100, 1000) == pytest.approx(10)
    assert m.attainable_tflops(1000, 100, 1000) == pytest.approx(100)
    assert m.bound(50, 100, 1000) == "memory" and m.bound(150, 100, 1000) == "compute"


def test_fp8_doubles_decode_intensity():
    assert m.decode_intensity(1, 1) == pytest.approx(2 * m.decode_intensity(1, 2))


def test_batch1_decode_memory_bound_everywhere():
    for g in specs.load():
        if g["fp16_dense_tflops"]:
            assert m.bound(m.decode_intensity(1), g["fp16_dense_tflops"], g["hbm_gbs"]) == "memory"


def test_long_prefill_compute_bound_on_l4_class():
    assert m.bound(m.prefill_intensity(2048), 121, 300) == "compute"
