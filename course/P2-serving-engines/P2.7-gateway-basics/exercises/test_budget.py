import importlib.util
import os
from pathlib import Path

import httpx

from s2s.gatewaytest import running_gateway
from s2s.mockserver import running_mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("budget_ut", HERE / ("solutions/budget.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "budget.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_sliding_window(clock):
    b = m.TokenBudget(limit=100, window_s=60, clock=clock)
    assert b.check(100) and b.remaining() == 100     # check() doesn't consume
    b.commit(70)
    assert b.remaining() == 30 and not b.check(31)
    clock.advance(30)
    b.commit(20)
    clock.advance(30)                                # the first commit is exactly 60 s old: expired
    assert b.remaining() == 80
    clock.advance(30)
    assert b.remaining() == 100


def test_gateway_enforces_budget():
    with running_mock(TIME_SCALE=0.1) as be:
        with running_gateway([be], tokens_per_window=60, window_s=3600) as gw:
            codes = []
            with httpx.Client() as c:
                for _ in range(6):
                    r = c.post(f"{gw}/v1/completions", headers={"Authorization": "Bearer k"},
                               json={"model": "mock-llama-8b", "prompt": "one two three", "max_tokens": 10})
                    codes.append(r.status_code)
    assert codes[0] == 200 and 429 in codes
