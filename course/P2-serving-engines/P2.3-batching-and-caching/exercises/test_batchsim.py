import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
path = HERE / "solutions/batchsim.py" if os.environ.get("S2S_SOLUTIONS") == "1" else HERE / "batchsim.py"
spec = importlib.util.spec_from_file_location("batchsim_under_test", path)
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def trace(seed, n=120, gap=30.0, long_prompt_every=0):
    rng, t, out = random.Random(seed), 0.0, []
    for i in range(n):
        t += rng.expovariate(1 / gap)
        p = 4000 if long_prompt_every and i % long_prompt_every == 0 else rng.choice([32, 128, 512])
        out.append(bs.Req(i, t, p, rng.choice([4, 16, 64, 256])))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_continuous_beats_static(seed):
    reqs = trace(seed)
    c, s = bs.continuous(reqs, 8), bs.static(reqs, 8)
    assert np.mean(list(c.e2e(reqs).values())) < np.mean(list(s.e2e(reqs).values()))


def test_invariants():
    reqs = trace(5)
    o = bs.continuous(reqs, 8)
    assert max(n for _, _, n, _ in o.steps) <= 8
    for r in reqs:
        assert len(o.token_times[r.rid]) == r.output
        assert o.first[r.rid] >= r.arrival
        assert o.token_times[r.rid] == sorted(o.token_times[r.rid])


def test_chunked_prefill_bounds_itl():
    reqs = trace(7, n=150, gap=15.0, long_prompt_every=25)
    cost = bs.Cost()
    plain, chunked = bs.continuous(reqs, 32, cost), bs.continuous(reqs, 32, cost, chunk_tokens=512)
    worst = lambda o: max(o.max_itl(r.rid) for r in reqs)  # noqa: E731
    bound = cost.base_ms + cost.per_seq_ms * 32 + cost.prefill_ms_per_token * 512
    assert worst(chunked) <= bound + 1e-6
    assert worst(chunked) < worst(plain)
    long_ids = [r.rid for r in reqs if r.prompt == 4000]
    assert np.mean([chunked.ttft(reqs)[i] for i in long_ids]) > np.mean([plain.ttft(reqs)[i] for i in long_ids])
