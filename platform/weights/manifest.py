"""manifest.py — a weight registry manifest: every file of a model version, with its size and sha256.

Registry layout (S3 or any directory):
    <root>/<model>/<version>/manifest.json
    <root>/<model>/<version>/<files…>          (*.safetensors shards, model.safetensors.index.json, config.json, tokenizer…)
Versions are immutable: a new upload is a new version, and "latest" is a pointer file, never an overwrite.

    python -m weights.manifest build  <dir> --model NAME --version V    # writes <dir>/manifest.json
    python -m weights.manifest verify <dir>                             # exit 1 on any mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CHUNK = 8 << 20
MANIFEST = "manifest.json"


@dataclass
class Entry:
    path: str
    size: int
    sha256: str


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def build(root: Path, model: str, version: str) -> dict:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST)
    entries = [Entry(str(p.relative_to(root)), p.stat().st_size, sha256_file(p)) for p in files]
    return {"model": model, "version": version, "total_bytes": sum(e.size for e in entries),
            "files": [asdict(e) for e in entries]}


def verify(root: Path, manifest: dict | None = None) -> list[str]:
    """Return a list of problems (empty = OK): missing, wrong size, wrong hash, unexpected extra files."""
    manifest = manifest or json.loads((root / MANIFEST).read_text())
    problems, listed = [], set()
    for e in manifest["files"]:
        p = root / e["path"]
        listed.add(e["path"])
        if not p.is_file():
            problems.append(f"missing: {e['path']}")
        elif p.stat().st_size != e["size"]:                       # cheap check first
            problems.append(f"size: {e['path']} {p.stat().st_size} != {e['size']}")
        elif sha256_file(p) != e["sha256"]:
            problems.append(f"sha256: {e['path']}")
    for p in root.rglob("*"):
        rel = str(p.relative_to(root))
        if p.is_file() and p.name != MANIFEST and rel not in listed:
            problems.append(f"unexpected: {rel}")
    return problems


def safetensors_shards(root: Path) -> list[str]:
    """Shards named by model.safetensors.index.json (weight_map values), or the single model.safetensors."""
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        return sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    return ["model.safetensors"] if (root / "model.safetensors").exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("dir"); b.add_argument("--model", required=True); b.add_argument("--version", required=True)  # noqa: E702
    v = sub.add_parser("verify"); v.add_argument("dir")  # noqa: E702
    a = ap.parse_args(argv)
    root = Path(a.dir)
    if a.cmd == "build":
        (root / MANIFEST).write_text(json.dumps(build(root, a.model, a.version), indent=2))
        return 0
    probs = verify(root)
    print("\n".join(probs) or "OK")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
