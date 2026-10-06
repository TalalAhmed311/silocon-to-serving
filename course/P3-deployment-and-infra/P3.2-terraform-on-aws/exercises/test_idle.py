import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("idle", ROOT / "infra/aws/scripts/idle_decider.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def run(trace, limit=30):
    """trace: list of (gpu_util, ssm_sessions) per minute -> index of the minute that triggers stop, or None."""
    n = 0
    for i, (u, s) in enumerate(trace):
        n, stop = m.decide(n, u, s, limit)
        if stop:
            return i
    return None


def test_stops_after_exactly_limit_idle_minutes():
    assert run([(90, 0)] * 10 + [(0, 0)] * 40) == 10 + 30 - 1


def test_ssm_session_keeps_it_alive():
    assert run([(0, 1)] * 100) is None


def test_broken_nvidia_smi_counts_as_idle():
    assert run([(None, 0)] * 30) == 29


def test_busy_minute_resets():
    assert run([(0, 0)] * 29 + [(50, 0)] + [(0, 0)] * 29) is None
