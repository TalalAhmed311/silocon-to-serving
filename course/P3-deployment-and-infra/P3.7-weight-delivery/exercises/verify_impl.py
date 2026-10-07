"""Your manifest verifier. Same contract as platform/weights/manifest.py:verify (see 01-verifier.md)."""
from __future__ import annotations

from pathlib import Path


def verify(root: Path, manifest: dict) -> list[str]:
    return []  # TODO: missing / size / sha256 / unexpected
