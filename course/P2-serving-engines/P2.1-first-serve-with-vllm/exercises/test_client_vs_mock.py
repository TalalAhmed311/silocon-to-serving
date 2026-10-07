"""T0: your measure_e2e() against mockllm, whose cost model gives the expected latency in closed form."""
import importlib.util
import os
from pathlib import Path

import pytest

from s2s.mockserver import running_mock

HERE = Path(__file__).resolve().parent
path = HERE / "solutions" / "my_latency.py" if os.environ.get("S2S_SOLUTIONS") == "1" else HERE / "my_latency.py"
spec = importlib.util.spec_from_file_location("my_latency", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_e2e_matches_cost_model():
    base, per_seq, prefill = 8.0, 0.25, 0.05            # ms — mockllm cost model parameters
    with running_mock(BASE_STEP_MS=base, PER_SEQ_MS=per_seq, PREFILL_MS_PER_TOKEN=prefill, TIME_SCALE=1.0) as url:
        got = m.measure_e2e(url, "mock", prompt_tokens=128, output_tokens=32, n=5)
    # one sequence: first step includes prefill of 128 tokens, then 31 more steps
    want = ((base + per_seq + prefill * 128) + 31 * (base + per_seq)) / 1e3
    assert got == pytest.approx(want, rel=0.25)          # asyncio.sleep + HTTP overhead: generous tolerance
