"""Engine vs NumPy reference. Needs the engine binary (build it first, see platform/engine/v0/README.md).

Run: uv run pytest platform/engine/v0/tests
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
V0 = Path(__file__).resolve().parents[1]
# S2S_ENGINE lets the P0.5 exercises point this test at an engine built with the learner's ops.hpp.
ENGINE = Path(os.environ.get("S2S_ENGINE", ROOT / "build" / "engine-v0" / "s2s-engine"))
pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="build the engine first: cmake -S platform/engine/v0 -B build/engine-v0 && cmake --build build/engine-v0")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    subprocess.run([sys.executable, str(V0 / "tools" / "make_tiny_llama.py"), str(d)], check=True, capture_output=True)
    return d


def reference(model_dir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("llama_numpy", V0 / "reference" / "llama_numpy.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.LlamaNumpy(model_dir)


def test_teacher_forced_logits_match(tiny, tmp_path):
    ids = [5, 17, 3, 200, 42, 42, 7, 1, 99, 128, 64, 2]
    out = tmp_path / "logits.bin"
    subprocess.run([str(ENGINE), "--model", str(tiny), "--teacher-forced", " ".join(map(str, ids)),
                    "--dump-logits", str(out), "--threads", "2"], check=True)
    ref = reference(tiny)
    want = np.stack([ref.forward(t, p) for p, t in enumerate(ids)])
    got = np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)
    # fp32, different summation orders (SIMD + threads vs BLAS): 1e-4 relative on logits of magnitude ~1-10.
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_greedy_tokens_match(tiny):
    prompt = [1, 2, 3, 4]
    r = subprocess.run([str(ENGINE), "--model", str(tiny), "--prompt-ids", " ".join(map(str, prompt)), "--steps", "48"],
                       check=True, capture_output=True, text=True)
    got = [int(t) for t in r.stdout.split()]
    want, logits = reference(tiny).generate_greedy(prompt, 48)
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            # Allowed only if the reference itself had a near-tie at that step (fp32 summation-order noise).
            lg = logits[i - 1]
            top2 = np.sort(lg)[-2:]
            assert top2[1] - top2[0] < 1e-4, f"token {i}: engine {g} vs reference {w}, gap {top2[1] - top2[0]:.2e}"
            break  # after a legitimate tie-break the sequences diverge; nothing more to compare
