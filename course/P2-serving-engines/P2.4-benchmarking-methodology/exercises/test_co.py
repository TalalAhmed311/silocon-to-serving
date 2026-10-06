import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from s2s.mockserver import running_mock

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("cvo", ROOT / "course/P2-serving-engines/P2.4-benchmarking-methodology/examples/01_closed_vs_open.py")
cvo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvo)


@pytest.mark.slow
def test_open_loop_tail_exceeds_closed_loop():
    with running_mock(MAX_NUM_SEQS=8, BASE_STEP_MS=10, PER_SEQ_MS=0.5, TIME_SCALE=0.5) as url:
        closed = asyncio.run(cvo.closed_loop(url, users=32, duration=8, out_tokens=16))
        lam = len(closed) / 8
        w = cvo.Workload(rate=lam, duration_s=8, prompt_mean=64, prompt_sigma=0.01, output_mean=16, output_sigma=0.01, seed=3)
        opened = asyncio.run(cvo.run(w, url))
    p99 = lambda rs: np.percentile([r.ttft for r in rs if r.ttft is not None], 99)  # noqa: E731
    assert p99(opened) > p99(closed)
