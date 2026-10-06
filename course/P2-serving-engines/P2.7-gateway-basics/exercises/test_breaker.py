import importlib.util
import os
from pathlib import Path

import httpx
import pytest

from s2s.gatewaytest import running_gateway
from s2s.mockserver import running_mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("breaker_ut", HERE / ("solutions/breaker.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "breaker.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_state_machine(clock):
    b = m.CircuitBreaker(threshold=3, cooldown_s=10, clock=clock)
    for _ in range(3):
        assert b.allow()
        b.record(False)
    assert b.state == "open" and not b.allow()
    clock.advance(10)
    assert b.state == "half_open"
    assert b.allow() and not b.allow()        # one probe only
    b.record(False)
    assert b.state == "open"
    clock.advance(10)
    assert b.allow()
    b.record(True)
    assert b.state == "closed" and b.allow()


def test_fallback_end_to_end():
    with running_mock(FAIL_RATE=1.0, TIME_SCALE=0.1) as bad, running_mock(TIME_SCALE=0.1) as good:
        with running_gateway([bad, good]) as gw:
            with httpx.Client() as c:
                for _ in range(30):
                    r = c.post(f"{gw}/v1/completions", headers={"Authorization": "Bearer k"},
                               json={"model": "mock-llama-8b", "prompt": "hi", "max_tokens": 2})
                    assert r.status_code == 200, r.text
