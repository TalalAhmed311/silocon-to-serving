"""config.py — gateway configuration (YAML). Secrets are referenced by environment-variable NAME, never inlined."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Backend:
    name: str
    url: str
    models: list[str]
    api_key_env: str | None = None
    weight: float = 1.0
    timeout_s: float = 300.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class Tenant:
    name: str
    key_sha256: str                 # store hashes, not keys: a leaked config doesn't leak credentials
    previous_key_sha256: list[str] = field(default_factory=list)   # still accepted during a key rotation (P3.8)
    rps: float = 5.0
    burst: float = 10.0
    tokens_per_window: int = 1_000_000
    window_s: float = 86_400.0
    models: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    backends: list[Backend]
    tenants: list[Tenant]
    max_attempts: int = 3
    retry_budget_ratio: float = 0.2
    routing: str = "least_outstanding"   # or "weighted"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = yaml.safe_load(Path(path).read_text())
        return cls(backends=[Backend(**b) for b in d["backends"]], tenants=[Tenant(**t) for t in d["tenants"]],
                   **{k: v for k, v in d.items() if k not in ("backends", "tenants")})


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()
