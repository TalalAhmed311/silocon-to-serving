"""TorchRunner (+ CUDA graphs) vs NumpyPagedRunner on the tiny model. T2 — TODO(run-on: T2)."""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
V0 = Path(__file__).resolve().parents[2] / "v0"

from s2s_engine import LLMEngine, SamplingParams, SchedulerConfig  # noqa: E402
from s2s_engine.model_runner import NumpyPagedRunner, TorchRunner  # noqa: E402

PROMPTS = [[5, 17, 3, 200, 42], [1], [64, 2, 128, 7, 7, 7, 7, 7, 3], [9] * 30]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    subprocess.run([sys.executable, str(V0 / "tools" / "make_tiny_llama.py"), str(d)], check=True, capture_output=True)
    return d


def run(runner, nb, bs, reserve0=False):
    eng = LLMEngine(runner, nb, bs, SchedulerConfig(4, 16))
    if reserve0:
        eng.blocks._take()
    return eng.generate(PROMPTS, SamplingParams(max_tokens=8, ignore_eos=True))


@pytest.mark.parametrize("graphs", [False, True])
def test_fp32_greedy_matches_numpy(tiny, graphs):
    ref = run(NumpyPagedRunner(tiny, 64, 4), 64, 4)
    r = TorchRunner(tiny, 64, 4, dtype="float32", max_blocks_per_seq=16)
    if graphs:
        from s2s_engine.cuda_graph import capture_decode
        capture_decode(r, batch_sizes=(1, 2, 4))
    assert run(r, 64, 4, reserve0=graphs) == ref
