"""Repo-wide pytest setup: makes course/common/python importable as `s2s` and platform/ packages importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "course" / "common" / "python", ROOT / "platform"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
