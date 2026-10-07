"""batchsim.py — reference solution (static is identical to the starter; continuous is implemented)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("batchsim_starter", Path(__file__).resolve().parents[1] / "batchsim.py")
_starter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_starter)
Req, Cost, Outcome, static = _starter.Req, _starter.Cost, _starter.Outcome, _starter.static


def continuous(reqs, max_batch, cost=Cost(), chunk_tokens=None):
    out, t = Outcome(), 0.0
    waiting = sorted(reqs, key=lambda r: r.arrival)
    prefill_left: dict[int, int] = {}   # rid -> prompt tokens still to prefill
    decode_left: dict[int, int] = {}    # rid -> tokens still to emit (after the first)
    running: list = []                  # admitted, in admission order
    while waiting or running:
        if not running and waiting and waiting[0].arrival > t:
            t = waiting[0].arrival      # idle until the next arrival
        while waiting and waiting[0].arrival <= t and len(running) < max_batch:
            r = waiting.pop(0)
            running.append(r)
            prefill_left[r.rid] = r.prompt
        decoding = [r for r in running if prefill_left[r.rid] == 0]
        prefilling = [r for r in running if prefill_left[r.rid] > 0]
        budget = None if chunk_tokens is None else max(0, chunk_tokens - len(decoding))
        work: dict[int, int] = {}
        for r in prefilling:            # FIFO: earliest admitted prefills first
            n = prefill_left[r.rid] if budget is None else min(prefill_left[r.rid], budget)
            if n == 0:
                break
            work[r.rid] = n
            if budget is not None:
                budget -= n
        ptoks = sum(work.values())
        step = cost.base_ms + cost.per_seq_ms * len(decoding) + cost.prefill_ms_per_token * ptoks
        out.steps.append((t, t + step, len(decoding), ptoks))
        t += step
        for r in decoding:
            out.token_times[r.rid].append(t)
            decode_left[r.rid] -= 1
        for rid, n in work.items():
            prefill_left[rid] -= n
            if prefill_left[rid] == 0:  # prefill complete: this step produced the first token
                r = next(x for x in running if x.rid == rid)
                out.first[rid] = t
                out.token_times[rid] = [t]
                decode_left[rid] = r.output - 1
        for r in [r for r in running if prefill_left[r.rid] == 0 and decode_left.get(r.rid, 1) == 0]:
            out.done[r.rid] = out.token_times[r.rid][-1]
            running.remove(r)           # leaves immediately: its slot is reused next step
    return out
