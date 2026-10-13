import pytest

from s2s.exercise import load_impl

m = load_impl(__file__, "cost")

# Made-up numbers for testing only. They are not real prices. Look up the real price for your region.
EXAMPLE_PRICE = 2.0          # per hour
EXAMPLE_GB_MONTH = 0.10      # per GB-month


def test_session_cost():
    assert m.session_cost(EXAMPLE_PRICE, 3) == pytest.approx(6.0)
    assert m.session_cost(EXAMPLE_PRICE, 3, sessions=4) == pytest.approx(24.0)
    assert m.session_cost(EXAMPLE_PRICE, 0) == 0


def test_idle_cost_is_pro_rata():
    assert m.idle_cost(100, EXAMPLE_GB_MONTH, 30) == pytest.approx(10.0)
    assert m.idle_cost(100, EXAMPLE_GB_MONTH, 15) == pytest.approx(5.0)
    assert m.idle_cost(0, EXAMPLE_GB_MONTH, 30) == 0


def test_forgotten_instance_runs_24h_a_day():
    assert m.forgotten_cost(EXAMPLE_PRICE, 1) == pytest.approx(48.0)
    assert m.forgotten_cost(EXAMPLE_PRICE, 3) == pytest.approx(144.0)


def test_one_forgotten_weekend_dwarfs_the_session():
    # a 3 h session vs forgetting the instance for 2 days
    assert m.forgotten_cost(EXAMPLE_PRICE, 2) > 10 * m.session_cost(EXAMPLE_PRICE, 3)


def test_budget_ok_keeps_headroom():
    assert m.budget_ok(80, 100) is True            # exactly 20% headroom is allowed
    assert m.budget_ok(80.01, 100) is False
    assert m.budget_ok(95, 100, headroom=0.0) is True
    assert m.budget_ok(101, 100, headroom=0.0) is False
