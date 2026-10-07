import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "flops")
ROOT = Path(__file__).resolve().parents[5]


def counted():
    spec = importlib.util.spec_from_file_location(
        "counted", ROOT / "course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/01_numpy_llama_counted.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("t", [1, 10, 100])
def test_vs_instrumented_counter(tmp_path, t):
    subprocess.run([sys.executable, str(ROOT / "platform/engine/v0/tools/make_tiny_llama.py"), str(tmp_path), "--max-seq", "128"],
                   check=True, capture_output=True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    model = counted().Counted(tmp_path, 2.0)
    for p in range(t - 1):
        model.forward(1, p)
    model.flops.clear()
    model.forward(1, t - 1)                      # attends to t positions
    measured = sum(model.flops.values())
    # 3%: the counter also tallies norms, RoPE and SiLU, which are ~2% of a model this tiny (d = 64).
    assert abs(m.decode_flops(cfg, t) - measured) <= 3e-2 * measured
