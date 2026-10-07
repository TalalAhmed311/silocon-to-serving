"""Exercise 1: MIG layout for a tenant mix (T0). Profile tables are UNVERIFIED inputs; the tests check the packing."""
import importlib.util
import os
from pathlib import Path

import pytest

from partitioning import mig

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "layout_ut", HERE / ("solutions/layout.py" if os.environ.get("S2S_SOLUTIONS") == "1" else "layout.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
T = mig.Tenant


def slices(gpu, layout):
    return [(sum(mig.PROFILES[gpu][p][0] for _, p in g), sum(mig.PROFILES[gpu][p][1] for _, p in g)) for g in layout]


def test_every_tenant_placed_once_with_enough_memory():
    ts = [T("a", 35), T("b", 8), T("c", 15), T("d", 9), T("e", 70)]
    layout = m.plan("A100-80GB", ts)
    placed = {n: p for g in layout for n, p in g}
    assert sorted(placed) == ["a", "b", "c", "d", "e"]
    for t in ts:
        assert mig.PROFILES["A100-80GB"][placed[t.name]][2] >= t.mem_gb
    for c, mem in slices("A100-80GB", layout):
        assert c <= 7 and mem <= 8


def test_seven_small_tenants_share_one_gpu():
    layout = m.plan("A100-80GB", [T(f"t{i}", 9) for i in range(7)])
    assert len(layout) == 1 and all(p == "1g.10gb" for _, p in layout[0])


MIX = [T("x", 40), T("y", 40), T("p", 20), T("q", 20)] + [T(f"s{i}", 10) for i in range(4)]
# slices: compute 3+3+2+2+1+1+1+1 = 14, memory 4+4+2+2+1+1+1+1 = 16 → lower bound 2 GPUs, and 2 is achievable:
# {x, p, q} = 7c/8m and {y, s0..s3} = 7c/8m. First-fit-decreasing by memory puts x and y together (6c/8m) and needs 3.


def test_ffd_is_valid_but_not_optimal():
    layout = m.plan("A100-80GB", MIX)
    assert 2 <= len(layout) <= 3
    for c, mem in slices("A100-80GB", layout):
        assert c <= 7 and mem <= 8


@pytest.mark.xfail(reason="exercise 1, part 3: beat first-fit decreasing (exact search or better ordering)", strict=False)
def test_optimal_two_gpus():
    assert len(m.plan("A100-80GB", MIX)) == 2


def test_min_compute_respected():
    layout = m.plan("A100-80GB", [T("hot", 10, min_compute=3)])
    assert layout[0][0][1] in ("3g.40gb", "4g.40gb", "7g.80gb")


def test_impossible_tenant_raises():
    with pytest.raises(ValueError):
        m.plan("A100-80GB", [T("huge", 120)])
