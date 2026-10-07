import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import parallel  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tp_mlp_ut", HERE / ("solutions/tp_mlp.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "tp_mlp.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def weights(d=64, f=256, seed=0):
    rng = np.random.default_rng(seed)
    s = 1 / np.sqrt(d)
    return (rng.standard_normal((d, f)).astype(np.float32) * s, rng.standard_normal((d, f)).astype(np.float32) * s,
            rng.standard_normal((f, d)).astype(np.float32) / np.sqrt(f))


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 17])
def test_tp_matches_unsplit_fp32(tp, tokens):
    wg, wu, wd = weights()
    x = np.random.default_rng(1).standard_normal((tokens, 64)).astype(np.float32)
    want = parallel.mlp(x, wg, wu, wd)
    partials, out = m.tp_mlp(x, wg, wu, wd, tp)
    assert len(partials) == tp and all(p.shape == want.shape for p in partials)
    np.testing.assert_allclose(out, want, atol=1e-5, rtol=1e-5)


def test_partials_alone_are_not_the_answer():
    wg, wu, wd = weights()
    x = np.random.default_rng(2).standard_normal((3, 64)).astype(np.float32)
    partials, out = m.tp_mlp(x, wg, wu, wd, 4)
    assert not np.allclose(partials[0], out, atol=1e-3)     # without the all-reduce, each rank is wrong


def test_wrong_split_order_is_wrong():
    wg, wu, wd = weights()
    x = np.random.default_rng(3).standard_normal((3, 64)).astype(np.float32)
    assert not np.allclose(parallel.tp_mlp_wrong_order(x, wg, wu, wd, 4), parallel.mlp(x, wg, wu, wd), atol=1e-3)
