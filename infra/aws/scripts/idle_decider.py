"""idle_decider.py — the idle-watchdog decision as a pure function (mirrors user_data.sh.tftpl), so it can be tested.

decide(state_minutes, gpu_util_pct, ssm_sessions, idle_limit) -> (new_state_minutes, stop: bool)
"""


def decide(state_minutes: int, gpu_util_pct: float | None, ssm_sessions: int, idle_limit: int) -> tuple[int, bool]:
    util = 0.0 if gpu_util_pct is None else gpu_util_pct          # nvidia-smi missing/failing counts as idle
    n = state_minutes + 1 if (util < 5 and ssm_sessions == 0) else 0
    return n, n >= idle_limit
