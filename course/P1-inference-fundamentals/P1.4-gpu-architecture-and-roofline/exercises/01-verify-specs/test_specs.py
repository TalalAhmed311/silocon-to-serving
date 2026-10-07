import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import specs  # noqa: E402


def test_loads_and_every_entry_sourced():
    gpus = specs.load()
    assert len(gpus) >= 5
    assert all(g["source"] for g in gpus)


def test_ridges_are_sane():
    for g in specs.load():
        r = specs.ridge(g)
        if r is not None:
            assert 10 < r < 2000, f"{g['name']}: ridge {r:.0f} looks wrong (TFLOPS vs GB/s units?)"


def test_sparse_numbers_not_used():
    # Dense fp8 should be ~2x dense fp16 where both exist; a 4x ratio usually means a sparse figure slipped in.
    for g in specs.load():
        if g.get("fp8_dense_tflops") and g.get("fp16_dense_tflops"):
            assert 1.5 < g["fp8_dense_tflops"] / g["fp16_dense_tflops"] < 2.5, g["name"]
