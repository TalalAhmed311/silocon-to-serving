"""Repo-wide pytest setup: makes course/common/python importable as `s2s` and platform/ packages importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "course" / "common" / "python", ROOT / "platform"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_collection_modifyitems(config, items):
    """`slow` tests run only when selected explicitly (e.g. `pytest -m slow`), so the default T0 run stays quick."""
    import pytest
    if "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
