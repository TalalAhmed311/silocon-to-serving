import importlib.util
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "platform"))
from specdec.core import expected_tokens_per_round  # noqa: E402

path = HERE / "solutions/formula.py" if os.environ.get("S2S_SOLUTIONS") == "1" else HERE / "formula.py"
spec = importlib.util.spec_from_file_location("formula_ut", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("alpha", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("k", [1, 4, 8])
def test_matches_formula(alpha, k):
    assert m.simulate(alpha, k) == pytest.approx(expected_tokens_per_round(alpha, k), rel=0.03)
