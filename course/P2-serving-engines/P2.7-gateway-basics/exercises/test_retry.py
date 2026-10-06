import importlib.util
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("retry_ut", HERE / ("solutions/retrypol.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "retrypol.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rules():
    b = m.RetryBudget(ratio=1.0, min_retries=100)
    assert not m.should_retry(503, None, True, 0, 3, b)       # streaming started
    assert not m.should_retry(400, None, False, 0, 3, b)
    assert m.should_retry(503, None, False, 0, 3, b)
    assert m.should_retry(None, ConnectionError(), False, 0, 3, b)
    assert not m.should_retry(503, None, False, 2, 3, b)      # third attempt was the last


def test_budget_caps_retries():
    b = m.RetryBudget(ratio=0.2, min_retries=0)
    for _ in range(1000):
        b.on_request()
    assert sum(b.try_spend() for _ in range(1000)) == 200


def test_backoff_jitter_bounds():
    rng = random.Random(0)
    xs = [m.backoff_s(3, base=0.1, cap=2.0, rng=rng) for _ in range(200)]
    assert all(0 <= x <= 0.8 for x in xs) and len(set(round(x, 6) for x in xs)) > 100
    assert all(m.backoff_s(10, base=0.1, cap=2.0, rng=rng) <= 2.0 for _ in range(50))
