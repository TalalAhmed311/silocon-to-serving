"""Your audit-log verifier. Contract: platform/tenancy/audit.py:verify (see 03-audit.md)."""
from __future__ import annotations

from pathlib import Path

from tenancy.audit import GENESIS, chain_hash  # noqa: F401  (use these)


def verify(path: str | Path, key: bytes | None = None, checkpoint: str | None = None) -> list[str]:
    return []  # TODO
