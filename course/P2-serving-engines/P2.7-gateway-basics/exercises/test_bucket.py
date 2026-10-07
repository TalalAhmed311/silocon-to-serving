import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bucket_ut", HERE / ("solutions/bucket.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "bucket.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_burst_then_refuse(clock):
    b = m.TokenBucket(rate=2, burst=5, clock=clock)
    assert all(b.allow() for _ in range(5))
    assert not b.allow()


def test_refill_and_cap(clock):
    b = m.TokenBucket(rate=2, burst=5, clock=clock)
    for _ in range(5):
        b.allow()
    clock.advance(1.0)                       # +2 tokens
    assert b.allow() and b.allow() and not b.allow()
    clock.advance(100)                       # capped at burst
    assert sum(b.allow() for _ in range(10)) == 5


def test_retry_after(clock):
    b = m.TokenBucket(rate=4, burst=1, clock=clock)
    assert b.allow()
    assert b.retry_after() == pytest.approx(0.25)
