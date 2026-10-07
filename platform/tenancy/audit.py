"""audit.py — a tamper-evident, append-only audit log (hash chain).

Each JSONL record carries `prev` (the previous record's hash) and `hash` = HMAC-SHA256(key, prev || canonical(record)).
Editing, deleting or reordering any record breaks every later hash; truncating the tail is caught by comparing the
last hash with a checkpoint stored somewhere else (an S3 object with Object Lock, CloudWatch Logs, a ticket).

The HMAC key comes from Secrets Manager/SSM at runtime (env var S2S_AUDIT_KEY in the pod via the CSI driver or an
init step) — never from the repo. Without a key it degrades to plain SHA-256: still detects edits by someone who
doesn't recompute the chain, but not by an attacker who can rewrite the whole file.

What gets logged: who (tenant, key hash prefix), what (action, model, route), outcome, token counts. Never prompts
or completions (PII), never full keys.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

GENESIS = "0" * 64


def canonical(rec: dict) -> bytes:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def chain_hash(key: bytes | None, prev: str, rec: dict) -> str:
    msg = prev.encode() + canonical(rec)
    return hmac.new(key, msg, hashlib.sha256).hexdigest() if key else hashlib.sha256(msg).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path, key: bytes | None = None, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.key = key if key is not None else (os.environ.get("S2S_AUDIT_KEY", "").encode() or None)
        self._lock = threading.Lock()
        self.last = GENESIS
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.last = json.loads(line)["hash"]

    def append(self, **fields) -> str:
        with self._lock:
            rec = {"ts": self.clock(), **fields}
            h = chain_hash(self.key, self.last, rec)
            with open(self.path, "a") as f:
                f.write(json.dumps({"rec": rec, "prev": self.last, "hash": h}, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.last = h
            return h


def verify(path: str | Path, key: bytes | None = None, checkpoint: str | None = None) -> list[str]:
    """Problems found (empty = intact). `checkpoint` = a last-hash recorded elsewhere, to detect truncation."""
    problems, prev, seen = [], GENESIS, set()
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    for i, line in enumerate(lines, 1):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {i}: not JSON")
            break
        if e["prev"] != prev:
            problems.append(f"line {i}: broken link (deleted or reordered record before it)")
        if chain_hash(key, e["prev"], e["rec"]) != e["hash"]:
            problems.append(f"line {i}: hash mismatch (record edited)")
        seen.add(e["hash"])
        prev = e["hash"]
    if checkpoint and checkpoint not in seen:
        problems.append("checkpoint hash not found (log truncated or replaced)")
    return problems
