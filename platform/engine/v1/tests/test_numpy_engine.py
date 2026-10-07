"""#0 v1 with the real (tiny) Llama on the paged NumPy runner == v0's contiguous reference, token for token. T0.
Batching, chunking, preemption and prefix caching must not change a single greedy token."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("safetensors")
V0 = Path(__file__).resolve().parents[2] / "v0"

from s2s_engine import LLMEngine, SamplingParams, SchedulerConfig  # noqa: E402
from s2s_engine.model_runner import NumpyPagedRunner  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    subprocess.run([sys.executable, str(V0 / "tools" / "make_tiny_llama.py"), str(d)], check=True, capture_output=True)
    return d


def reference(model_dir, prompt, steps):
    spec = importlib.util.spec_from_file_location("llama_numpy_ref", V0 / "reference" / "llama_numpy.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    toks, _ = m.LlamaNumpy(model_dir).generate_greedy(prompt, steps)
    return toks[len(prompt):]


PROMPTS = [[5, 17, 3, 200, 42], [5, 17, 3, 200, 42, 9, 9], [1], [64, 2, 128, 7, 7, 7, 7, 7, 3]]


@pytest.mark.parametrize("num_blocks,block_size,budget", [(64, 4, 512), (12, 2, 3)])
def test_paged_engine_matches_v0_reference(tiny, num_blocks, block_size, budget):
    steps = 6
    eng = LLMEngine(NumpyPagedRunner(tiny, num_blocks, block_size), num_blocks, block_size,
                    SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=budget))
    outs = eng.generate(PROMPTS, SamplingParams(max_tokens=steps, ignore_eos=True))
    for p, o in zip(PROMPTS, outs):
        assert o == reference(tiny, p, steps)
