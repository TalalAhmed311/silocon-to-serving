"""Exercise 3: the Young/Daly optimal checkpoint interval vs a Monte-Carlo simulation (T0)."""
import pytest

from training import faults


def test_young_formula():
    assert faults.young(C=60, M=6 * 3600) == pytest.approx((2 * 60 * 6 * 3600) ** 0.5)
    assert faults.daly(60, 6 * 3600) == pytest.approx(faults.young(60, 6 * 3600), rel=0.05)   # C ≪ M: they agree


def test_simulated_optimum_is_near_young():
    C, M, work = 30.0, 3600.0, 20 * 3600.0
    tau_star = faults.young(C, M)                   # ≈ 465 s
    taus = [tau_star * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)]
    wall = [faults.simulate(work, t, C, M, R=60, seed=1, trials=60) for t in taus]
    best = taus[wall.index(min(wall))]
    assert best in (taus[1], taus[2], taus[3])      # the minimum is within 2× of τ*
    assert wall[2] < wall[0] and wall[2] < wall[4]  # too-frequent and too-rare both lose


def test_waste_model_minimum_matches_young():
    C, M = 20.0, 7200.0
    grid = [10 * i for i in range(1, 400)]
    best = min(grid, key=lambda t: faults.waste_fraction(t, C, M))
    assert best == pytest.approx(faults.young(C, M), rel=0.1)
