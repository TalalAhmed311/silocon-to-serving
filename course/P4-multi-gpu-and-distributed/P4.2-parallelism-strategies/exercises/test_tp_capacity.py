"""Exercise 3: TP communication in the D3 capacity calculator (synthetic link numbers — inputs, not claims)."""
import importlib.util
import os
from pathlib import Path

import pytest

from capacity import core

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "tp_comm_ut", HERE / ("solutions/tp_comm.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "tp_comm.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

MODEL = core.Model(name="llama3-8b-shape", hidden=4096, intermediate=14336, layers=32, heads=32, kv_heads=8, vocab=128256)


def test_matches_reference():
    for tp, tok, bw, a in [(2, 1, 50, 5), (4, 64, 25, 10), (8, 512, 300, 3)]:
        assert m.tp_comm_seconds_per_step(MODEL, tp, tok, bw, a) == pytest.approx(core.tp_comm_seconds_per_step(MODEL, tp, tok, bw, a))


def test_tp1_is_free_and_decode_is_latency_bound():
    assert m.tp_comm_seconds_per_step(MODEL, 1, 1, 50, 10) == 0
    lat_only = m.tp_comm_seconds_per_step(MODEL, 4, 1, 1e9, 10)          # infinite bandwidth
    full = m.tp_comm_seconds_per_step(MODEL, 4, 1, 50, 10)
    assert full == pytest.approx(lat_only, rel=0.05)                     # 8 KB messages: α dominates at batch 1


def test_plan_includes_comm():
    no = core.plan(MODEL, "L4", tp=4)
    yes = core.plan(MODEL, "L4", tp=4, link_gbs=25, link_alpha_us=10)
    for B in yes.decode_ceiling:
        assert yes.decode_ceiling[B][0] < no.decode_ceiling[B][0]
