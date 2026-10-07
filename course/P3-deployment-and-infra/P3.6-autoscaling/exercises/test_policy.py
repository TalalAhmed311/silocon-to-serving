"""P3.6 exercise 1: the scaling policy. S2S_SOLUTIONS=1 runs against platform/autoscaler/policy.py."""
import os
import random

import pytest

from s2s.exercise import load_impl

if os.environ.get("S2S_SOLUTIONS") == "1":
    from autoscaler import policy as m
else:
    m = load_impl(__file__, "policy_impl")


def run(trace, p, dt=15.0, replicas=0):
    s = m.State(replicas=replicas)
    return [m.step(s, i * dt, w, p) for i, w in enumerate(trace)]


def test_raw_desired():
    p = m.Policy(target_per_replica=24)
    assert [m.raw_desired(w, p) for w in (0, 1, 24, 25, 48, 49)] == [0, 1, 1, 2, 2, 3]


def test_from_zero_activates_and_clamps():
    p = m.Policy(target_per_replica=10, max_replicas=4, max_up_per_step=10)
    assert run([5], p) == [1]
    assert run([1000], p) == [4]


def test_scale_up_is_rate_limited():
    p = m.Policy(target_per_replica=10, max_up_per_step=2, max_replicas=10)
    assert run([100, 100, 100, 100], p) == [2, 4, 6, 8]


def test_scale_down_is_stabilized_and_rate_limited():
    p = m.Policy(target_per_replica=10, down_stabilization_s=60, max_down_per_step=1, max_replicas=10, cooldown_s=1e9)
    out = run([40] + [5] * 12, p, dt=15, replicas=4)
    assert out[:5] == [4, 4, 4, 4, 4]          # within 60 s of the 40-work sample, keep 4
    assert out[5] == 3                          # then step down by one at a time
    assert out[-1] == 1                         # floor of 1 while work > 0
    assert all(a - b <= 1 for a, b in zip(out, out[1:]))


def test_scale_to_zero_only_after_cooldown():
    p = m.Policy(target_per_replica=10, cooldown_s=600, down_stabilization_s=0)
    out = run([5] + [0] * 50, p, dt=15)
    t_zero = out.index(0) * 15
    assert t_zero >= 600
    assert out[-1] == 0


def test_min_replicas_respected():
    p = m.Policy(min_replicas=1, cooldown_s=0, down_stabilization_s=0)
    assert run([0, 0, 0], p, replicas=1) == [1, 1, 1]


def test_less_flapping_than_raw():
    rng = random.Random(0)
    trace = [max(0.0, 50 + rng.gauss(0, 15)) for _ in range(200)]
    p = m.Policy(target_per_replica=20, max_replicas=10, max_up_per_step=10)
    raw = [m.raw_desired(w, p) for w in trace]
    pol = run(trace, p)
    changes = lambda xs: sum(a != b for a, b in zip(xs, xs[1:]))  # noqa: E731
    assert changes(pol) * 2 < changes(raw), (changes(pol), changes(raw))
