"""C1 exercises 1, 2, 4 (T0)."""
import random

from failover.budgets import GCounter, RegionBudget
from failover.gameday import analyze
from failover.health import HealthTracker, Probe, flaps
from failover.policy import Controller, decide

INF = float("inf")


# ---- health with hysteresis --------------------------------------------------------------------------------------
def test_single_blips_do_not_flap():
    h = HealthTracker(fail_threshold=3, rise_threshold=5)
    for t in range(200):
        h.observe(Probe(t, ok=(t % 7 != 0)))                # one failure every 7 probes
    assert h.up and flaps(h.transitions) == 0


def test_outage_detected_within_threshold_and_recovery_is_slower():
    h = HealthTracker(fail_threshold=3, rise_threshold=5)
    for t in range(10):
        h.observe(Probe(t, ok=True))
    for t in range(10, 20):
        h.observe(Probe(t, ok=False))
    assert h.transitions == [(12, "down")]                   # third consecutive failure
    for t in range(20, 30):
        h.observe(Probe(t, ok=True))
    assert h.transitions[-1] == (24, "up")                  # fifth consecutive success


def test_slow_is_down():
    h = HealthTracker(fail_threshold=2, latency_slo_s=1.0)
    h.observe(Probe(0, ok=True, latency_s=5))
    h.observe(Probe(1, ok=True, latency_s=5))
    assert not h.up


def test_flapping_bounded_under_noise():
    rng = random.Random(0)
    h = HealthTracker(fail_threshold=3, rise_threshold=5)
    for t in range(10_000):
        h.observe(Probe(t, ok=rng.random() > 0.2))          # 20 % random loss, no real outage
    # P(3 consecutive failures) per probe ≈ 0.8·0.2³ ≈ 0.6 %: some downs happen; hysteresis keeps them rare
    assert flaps(h.transitions) < 200


# ---- decision ----------------------------------------------------------------------------------------------------
def test_decide_table():
    up = {"a": True, "b": True}
    assert decide("a", "a", "b", up, {"a": 0, "b": 0}, 1000, -INF).active == "a"
    assert decide("a", "a", "b", {"a": False, "b": True}, {"a": INF, "b": 0}, 1000, -INF).active == "b"
    assert decide("a", "a", "b", {"a": False, "b": False}, {"a": INF, "b": INF}, 1000, -INF).reason == "both down: hold"
    assert decide("a", "a", "b", {"a": False, "b": True}, {"a": INF, "b": 0}, 1000, 950, min_dwell_s=120).active == "a"
    # failback needs the primary stable for failback_after_s
    assert decide("b", "a", "b", up, {"a": 900, "b": 0}, 1000, -INF, failback_after_s=600).active == "b"
    assert decide("b", "a", "b", up, {"a": 300, "b": 0}, 1000, -INF, failback_after_s=600).active == "a"
    assert decide("b", "a", "b", up, {"a": 0, "b": 0}, 1000, -INF, auto_failback=False).active == "b"


def test_controller_no_ping_pong():
    c = Controller("a", "b", failback_after_s=300, min_dwell_s=60)
    t = 0
    for t in range(0, 100, 10):
        c.step(t, {"a": True, "b": True})
    for t in range(100, 400, 10):
        c.step(t, {"a": (t // 10) % 2 == 0, "b": True})   # primary health oscillates every 10 s
    assert len(c.log) <= 300 // 60 + 1
    for t in range(400, 1200, 10):
        c.step(t, {"a": True, "b": True})
    assert c.active == "a" and c.log[-1][3].endswith("fail back")


# ---- budgets -----------------------------------------------------------------------------------------------------
def test_gcounter_merge_is_order_independent():
    rng = random.Random(1)
    regions = [GCounter(r) for r in "xyz"]
    for _ in range(300):
        g = rng.choice(regions)
        g.add(rng.choice("ab"), rng.randint(0, 50))
        if rng.random() < 0.3:
            rng.choice(regions).merge(rng.choice(regions))
    for _ in range(2):                                      # full exchange in two rounds converges
        for g in regions:
            for h in regions:
                g.merge(h)
    assert len({tuple(sorted(g.state().items())) for g in regions}) == 1


def simulate(split, seed=0, regions=3, budget=10_000, syncs_every=50):
    rng = random.Random(seed)
    counters = [GCounter(str(i)) for i in range(regions)]
    gates = [RegionBudget(c, {"t": budget}, regions, split=split) for c in counters]
    for step in range(3000):
        i = rng.randrange(regions)
        gates[i].admit("t", rng.randint(1, 40))
        if step % syncs_every == 0:
            for c in counters:
                for d in counters:
                    c.merge(d)
            for g in gates:
                g.on_sync()
    total = 0
    for c in counters:
        for d in counters:
            c.merge(d)
    total = counters[0].used("t")
    return total


def test_split_budget_never_overspends():
    for seed in range(10):
        assert simulate(split=True, seed=seed) <= 10_000


def test_naive_budget_can_overspend():
    assert max(simulate(split=False, seed=s) for s in range(10)) > 10_000


# ---- game-day timeline -----------------------------------------------------------------------------------------
def test_gameday_timeline():
    reqs = []
    for i in range(100):                                    # 1 req/s; region a dies at t=50, b takes over at t=80
        t = float(i)
        if t < 50:
            reqs.append({"t": t, "end": t + 0.5, "ok": True, "region": "a"})
        elif t < 80:
            reqs.append({"t": t, "end": t + 10, "ok": False, "region": None})
        else:
            reqs.append({"t": t, "end": t + 0.5, "ok": True, "region": "b"})
    r = analyze(reqs, kill_t=50, switch_t=75)
    assert r["old_region"] == "a" and r["failed"] == 30
    assert r["detection_s"] == 25 and r["first_success_new_region_s"] == 30.5
    assert r["failover_s"] == 99.5 - 50                      # 20th consecutive success in b ends at 99.5
