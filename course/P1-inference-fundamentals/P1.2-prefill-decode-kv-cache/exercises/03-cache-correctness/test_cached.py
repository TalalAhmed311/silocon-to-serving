import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "cached")
ROOT = Path(__file__).resolve().parents[5]
V0 = ROOT / "platform/engine/v0"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    subprocess.run([sys.executable, str(V0 / "tools/make_tiny_llama.py"), str(d)], check=True, capture_output=True)
    return d


def reference(d):
    spec = importlib.util.spec_from_file_location("llama_numpy", V0 / "reference/llama_numpy.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    return r.LlamaNumpy(d)


def test_prefill_then_decode_matches_reference(tiny):
    prompt, more = [4, 8, 15, 16, 23, 42], [7, 99, 3]
    ref = reference(tiny)
    want = [ref.forward(t, p) for p, t in enumerate(prompt + more)]
    model = m.PrefillThenDecode(tiny)
    np.testing.assert_allclose(model.prefill(prompt), want[len(prompt) - 1], atol=1e-4, rtol=1e-4)
    for i, t in enumerate(more):
        np.testing.assert_allclose(model.decode(t), want[len(prompt) + i], atol=1e-4, rtol=1e-4)


def test_no_future_leakage(tiny):
    a = m.PrefillThenDecode(tiny).prefill_all_logits([1, 2, 3, 4, 5])
    b = m.PrefillThenDecode(tiny).prefill_all_logits([1, 2, 3, 4, 200])   # only the last token differs
    np.testing.assert_allclose(a[:4], b[:4], atol=1e-6)
    assert np.abs(a[4] - b[4]).max() > 1e-3
