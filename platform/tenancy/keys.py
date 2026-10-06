"""keys.py — tenant API keys: generate, hash, rotate without downtime.

Rotation (no downtime, no coordinated client cut-over):
  1. new = generate();  config: key_sha256 = hash(new), previous_key_sha256 += [old hash]   → both keys work
  2. roll the gateway (≥ 2 replicas + readiness probe = no dropped requests); hand `new` to the tenant via the secret store
  3. watch gateway_requests_total by key generation (the gateway logs which hash matched) until the old key is unused
  4. drop the old hash from previous_key_sha256 and roll again
Keys are shown once, stored only as SHA-256 hashes in config, and delivered through Secrets Manager — never chat/email.
"""
from __future__ import annotations

import hashlib
import secrets

PREFIX = "s2s_"


def generate() -> str:
    return PREFIX + secrets.token_urlsafe(32)          # 256 bits; the prefix makes leaked keys greppable by scanners


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def start_rotation(tenant: dict, new_key: str) -> dict:
    """Return the tenant config with `new_key` primary and the old hash still accepted."""
    t = dict(tenant)
    t["previous_key_sha256"] = [*tenant.get("previous_key_sha256", []), tenant["key_sha256"]]
    t["key_sha256"] = key_hash(new_key)
    return t


def finish_rotation(tenant: dict) -> dict:
    t = dict(tenant)
    t["previous_key_sha256"] = []
    return t
