import random

import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "retry")

# (status, idempotent, expected) on attempt 1 with max_attempts=3
TABLE = [
    (200, True, False), (200, False, False),
    (400, True, False), (401, True, False), (404, True, False),
    (429, True, True), (429, False, True),
    (500, True, True), (500, False, False),
    (502, True, True), (502, False, False),
    (503, True, True), (503, False, True),
    (504, True, True), (504, False, False),
    (None, True, True), (None, False, False),
]


@pytest.mark.parametrize("status,idem,want", TABLE)
def test_table(status, idem, want):
    assert m.should_retry(status, 1, idem) is want


def test_attempt_limit():
    assert m.should_retry(503, 2, True, max_attempts=3) is True
    assert m.should_retry(503, 3, True, max_attempts=3) is False
    assert m.should_retry(429, 1, True, max_attempts=1) is False


def test_backoff_bound_doubles_then_caps():
    assert [m.backoff_bound(a) for a in (1, 2, 3, 4)] == pytest.approx([0.1, 0.2, 0.4, 0.8])
    assert m.backoff_bound(10) == 5.0                       # 0.1 * 512 = 51.2, capped
    assert m.backoff_bound(3, base=1.0, cap=3.0) == 3.0


def test_jitter_stays_inside_bound_and_varies():
    rng = random.Random(7)
    for attempt in (1, 2, 5, 12):
        vals = [m.backoff_delay(attempt, rng) for _ in range(200)]
        assert all(0.0 <= v <= m.backoff_bound(attempt) for v in vals)
        assert len(set(vals)) > 100
