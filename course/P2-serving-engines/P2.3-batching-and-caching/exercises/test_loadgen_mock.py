import asyncio
import sys
from pathlib import Path

import pytest

from s2s.mockserver import running_mock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform"))
from loadgen.analysis import find_knee, summarize  # noqa: E402
from loadgen.runner import run  # noqa: E402
from loadgen.workload import Workload  # noqa: E402


@pytest.mark.slow
def test_knee_near_closed_form():
    # scale 0.5: an 8 ms real step keeps asyncio sleep and HTTP overheads small relative to the modelled cost
    scale, max_seqs, base, per_seq, out_mean = 0.5, 16, 12.0, 0.25, 32
    # saturated step (all 16 running) in *mock time*; real time is ×scale
    step_s = (base + per_seq * max_seqs) / 1e3 * scale
    saturation_rps = max_seqs / (step_s * out_mean)          # requests/s the engine can complete
    rates = [saturation_rps * f for f in (0.25, 0.5, 0.75, 0.9, 1.1, 1.5)]
    with running_mock(TIME_SCALE=scale, MAX_NUM_SEQS=max_seqs, BASE_STEP_MS=base, PER_SEQ_MS=per_seq,
                      PREFILL_MS_PER_TOKEN=0.0) as url:
        rows = []
        for rate in rates:
            w = Workload(rate=rate, duration_s=6, prompt_mean=16, output_mean=out_mean, output_sigma=0.01, seed=1)
            res = asyncio.run(run(w, url))
            rows.append(summarize(res, 6, ttft_slo=20 * step_s, tpot_slo=1.0))
    knee = find_knee(rows, ttft_slo=20 * step_s)
    assert knee is not None
    assert 0.6 * saturation_rps <= knee["offered_rps"] <= 1.25 * saturation_rps
