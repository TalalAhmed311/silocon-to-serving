"""Load an exercise implementation: the learner's starter by default, the reference with S2S_SOLUTIONS=1.

Layout convention for every Python exercise:
    exercises/<NN-name>/<module>.py            starter (has TODOs)
    exercises/solutions/<NN-name>/<module>.py  reference solution
    exercises/<NN-name>/test_<module>.py       test, calls load_impl(__file__, "<module>")
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def load_impl(test_file: str, module: str) -> ModuleType:
    ex_dir = Path(test_file).resolve().parent
    if os.environ.get("S2S_SOLUTIONS") == "1":
        path = ex_dir.parent / "solutions" / ex_dir.name / f"{module}.py"
    else:
        path = ex_dir / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"{ex_dir.name}_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
