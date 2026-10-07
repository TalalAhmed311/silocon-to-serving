"""Exercise 2: the bubble formula vs the event simulation, and 1F1B's memory advantage."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel as P  # noqa: E402


@pytest.mark.parametrize("p,m", [(2, 1), (4, 4), (4, 16), (8, 32)])
@pytest.mark.parametrize("sched", ["gpipe", "1f1b"])
def test_bubble_formula_matches_simulation(p, m, sched):
    r = P.simulate_pipeline(p, m, t_fwd=1.0, t_bwd=2.0, schedule=sched)
    assert r.bubble == pytest.approx(P.bubble_fraction(p, m), abs=1e-9)


def test_more_microbatches_shrink_the_bubble():
    assert P.bubble_fraction(8, 64) < P.bubble_fraction(8, 8) < P.bubble_fraction(8, 1)


def test_1f1b_bounds_activation_memory():
    g = P.simulate_pipeline(4, 32, schedule="gpipe")
    o = P.simulate_pipeline(4, 32, schedule="1f1b")
    assert g.peak_inflight == 32          # GPipe stage 0 holds all microbatches' activations
    assert o.peak_inflight <= 4           # 1F1B: at most `stages` in flight
    assert o.makespan == pytest.approx(g.makespan)


def test_zero_memory_accounting():
    assert P.training_bytes_per_param("ddp", 8) == 16
    assert P.training_bytes_per_param("zero1", 8) == pytest.approx(4 + 1.5)
    assert P.training_bytes_per_param("fsdp", 8) == 2
