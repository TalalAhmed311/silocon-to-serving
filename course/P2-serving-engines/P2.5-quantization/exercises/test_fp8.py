import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
path = HERE / "solutions/fp8.py" if os.environ.get("S2S_SOLUTIONS") == "1" else HERE / "fp8.py"
spec = importlib.util.spec_from_file_location("fp8_ut", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_range_constants():
    v = m.representable("e4m3")
    assert v.max() == 448.0
    assert v[v > 0].min() == 2.0 ** -9
    assert m.representable("e5m2").max() == 57344.0


def test_matches_torch_e4m3fn():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8")
    v = m.representable("e4m3")
    mids = (v[1:] + v[:-1]) / 2
    rng = np.random.default_rng(0)
    x = np.concatenate([v, -v, mids, -mids, rng.uniform(-448, 448, 100_000)]).astype(np.float32)
    want = torch.from_numpy(x).to(torch.float8_e4m3fn).float().numpy()
    got = m.quantize(x, "e4m3")
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, f"{bad.size} mismatches, e.g. x={x[bad[:3]]} got={got[bad[:3]]} want={want[bad[:3]]}"


def test_saturates():
    assert m.quantize(np.array([1000.0, -1e6], np.float32), "e4m3").tolist() == [448.0, -448.0]
