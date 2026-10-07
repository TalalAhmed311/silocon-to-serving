"""cache.py — a node-local weight cache (NVMe instance store or a hostPath) with an LRU size limit.

    cache = WeightCache("/mnt/nvme/weights", max_bytes=200 << 30, store=DirStore("/registry") or S3Store("bucket/prefix"))
    path = cache.get("llama3-8b", "2025-01-rev3")     # local dir, verified against the manifest; fetched on a miss

Rules:
  * a version is fetched into a temp dir, verified, then renamed in: readers never see a partial copy
  * eviction removes least-recently-used versions until the new one fits; a version currently pinned is never evicted
  * an entry is "used" on every get(); recency is persisted in <root>/.lru.json so it survives pod restarts
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Protocol

from .manifest import MANIFEST, verify


class Store(Protocol):
    def manifest(self, model: str, version: str) -> dict: ...
    def fetch(self, model: str, version: str, rel: str, dst: Path) -> None: ...


class DirStore:
    """A registry on a filesystem path (T0 tests; or an EFS/FSx mount)."""
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def manifest(self, model, version):
        return json.loads((self.root / model / version / MANIFEST).read_text())

    def fetch(self, model, version, rel, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / model / version / rel, dst)


class S3Store:
    """A registry under s3://bucket/prefix (needs boto3; credentials from IRSA/Pod Identity or the instance role, never keys in env)."""
    def __init__(self, uri: str, max_concurrency: int = 16):
        import boto3
        from boto3.s3.transfer import TransferConfig
        self.bucket, _, self.prefix = uri.removeprefix("s3://").partition("/")
        self.s3 = boto3.client("s3")
        self.cfg = TransferConfig(max_concurrency=max_concurrency, multipart_chunksize=64 << 20)

    def _key(self, model, version, rel):
        return "/".join(x for x in (self.prefix, model, version, rel) if x)

    def manifest(self, model, version):
        return json.loads(self.s3.get_object(Bucket=self.bucket, Key=self._key(model, version, MANIFEST))["Body"].read())

    def fetch(self, model, version, rel, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, self._key(model, version, rel), str(dst), Config=self.cfg)


class CacheFull(Exception):
    pass


class WeightCache:
    def __init__(self, root: str | Path, max_bytes: int, store: Store, clock=time.time):
        self.root, self.max_bytes, self.store, self.clock = Path(root), max_bytes, store, clock
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pins: dict[str, int] = {}
        self.hits = self.misses = self.evictions = 0

    # --- bookkeeping -----------------------------------------------------------------------------
    def _lru_path(self) -> Path:
        return self.root / ".lru.json"

    def _lru(self) -> dict[str, float]:
        try:
            return json.loads(self._lru_path().read_text())
        except FileNotFoundError:
            return {}

    def _save_lru(self, d: dict[str, float]) -> None:
        tmp = self._lru_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, self._lru_path())

    @staticmethod
    def key(model: str, version: str) -> str:
        return f"{model}/{version}"

    def entry_bytes(self, key: str) -> int:
        return json.loads((self.root / key / MANIFEST).read_text())["total_bytes"]

    def used_bytes(self) -> int:
        return sum(self.entry_bytes(k) for k in self._lru() if (self.root / k).exists())

    # --- API -------------------------------------------------------------------------------------
    def pin(self, model, version):
        k = self.key(model, version)
        self._pins[k] = self._pins.get(k, 0) + 1

    def unpin(self, model, version):
        k = self.key(model, version)
        self._pins[k] -= 1
        if not self._pins[k]:
            del self._pins[k]

    def get(self, model: str, version: str) -> Path:
        k = self.key(model, version)
        with self._lock:
            lru = self._lru()
            if k in lru and (self.root / k).exists():
                self.hits += 1
                lru[k] = self.clock()
                self._save_lru(lru)
                return self.root / k
            self.misses += 1
            man = self.store.manifest(model, version)
            self._make_room(man["total_bytes"], lru)
            tmp = self.root / ".tmp" / uuid.uuid4().hex
            for e in man["files"]:
                self.store.fetch(model, version, e["path"], tmp / e["path"])
            (tmp / MANIFEST).write_text(json.dumps(man))
            problems = verify(tmp, man)
            if problems:
                shutil.rmtree(tmp, ignore_errors=True)
                raise IOError(f"{k} failed verification: {problems[:3]}")
            dst = self.root / k
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():                              # a stale copy the LRU file forgot about
                shutil.rmtree(dst)
            os.replace(tmp, dst)                         # atomic within one filesystem
            lru[k] = self.clock()
            self._save_lru(lru)
            return dst

    def _make_room(self, need: int, lru: dict[str, float]) -> None:
        present = {k: t for k, t in lru.items() if (self.root / k).exists()}
        sizes = {k: self.entry_bytes(k) for k in present}
        for k in plan_evictions(present, sizes, set(self._pins), need, self.max_bytes):
            shutil.rmtree(self.root / k)
            del lru[k]
            self.evictions += 1


def plan_evictions(lru: dict[str, float], sizes: dict[str, int], pinned: set[str], need: int, max_bytes: int) -> list[str]:
    """Keys to evict, least recently used first, so that used + need <= max_bytes. Pinned keys are never chosen.
    Raises CacheFull if it can't be done (need > max_bytes, or the pinned set alone leaves too little room)."""
    if need > max_bytes:
        raise CacheFull(f"{need} B > cache size {max_bytes} B")
    used = sum(sizes[k] for k in lru)
    out = []
    for k in sorted(lru, key=lru.get):
        if used + need <= max_bytes:
            break
        if k in pinned:
            continue
        used -= sizes[k]
        out.append(k)
    if used + need > max_bytes:
        raise CacheFull("not enough unpinned space")
    return out
