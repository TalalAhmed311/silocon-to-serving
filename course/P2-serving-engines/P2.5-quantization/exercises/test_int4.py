import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
path = HERE / "solutions/int4.py" if os.environ.get("S2S_SOLUTIONS") == "1" else HERE / "int4.py"
spec = importlib.util.spec_from_file_location("int4_ut", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rng = np.random.default_rng(0)


@pytest.mark.parametrize("symmetric", [True, False])
def test_error_within_half_scale(symmetric):
    w = rng.standard_normal((8, 256)).astype(np.float32)
    q, s, z, shape = m.quantize(w, 128, symmetric)
    err = np.abs(m.dequantize(q, s, z, shape) - w).reshape(8, 2, 128)
    assert np.all(err <= s / 2 + 1e-6)


def test_asymmetric_wins_on_skewed_data():
    w = np.abs(rng.standard_normal((4, 128))).astype(np.float32)
    e = lambda sym: np.abs(m.dequantize(*m.quantize(w, 128, sym)) - w).mean()  # noqa: E731
    assert e(False) <= e(True)


def test_smaller_groups_help_with_outliers():
    w = (0.02 * rng.standard_normal((16, 1024))).astype(np.float32)
    w[:, 7] = 1.0
    e = lambda g: np.abs(m.dequantize(*m.quantize(w, g, True)) - w).mean()  # noqa: E731
    assert e(32) < e(128) < e(1024)


def test_zero_group():
    w = np.zeros((1, 128), np.float32)
    assert np.all(m.dequantize(*m.quantize(w, 128, True)) == 0)


def test_pack_roundtrip():
    try:
        q = rng.integers(-8, 8, 257).astype(np.int8)
        assert np.array_equal(m.unpack(m.pack(q), 257), q)
    except NotImplementedError:
        pytest.skip("optional 1b not implemented")
