"""checkpoint.py — DCP checkpoints with an atomic "committed" marker and latest-checkpoint discovery.

Layout:  <root>/step_000120/ (DCP shards + metadata, written collectively by all ranks)
         <root>/step_000120/COMMITTED   (written by rank 0 AFTER every rank finished: a crash mid-save leaves no marker)
Resume:  latest_committed(root) → load → continue from step+1. Older checkpoints beyond `keep` are pruned by rank 0.

DCP (torch.distributed.checkpoint) saves each rank's shard of every DTensor/ShardedTensor and a global metadata file,
so a checkpoint saved with world size 2 loads into world size 4 (resharding happens at load: exercise 2).
On S3: write to local NVMe then sync (simple), or use an fsspec/S3 storage writer (UNVERIFIED for your torch version).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

STEP_RE = re.compile(r"step_(\d{6,})$")


def step_dir(root: str | Path, step: int) -> Path:
    return Path(root) / f"step_{step:06d}"


def committed_steps(root: str | Path) -> list[int]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in root.iterdir():
        m = STEP_RE.search(p.name)
        if m and (p / "COMMITTED").exists():
            out.append(int(m[1]))
    return sorted(out)


def latest_committed(root: str | Path) -> int | None:
    s = committed_steps(root)
    return s[-1] if s else None


def commit(root: str | Path, step: int) -> None:
    (step_dir(root, step) / "COMMITTED").write_text(str(step))


def prune(root: str | Path, keep: int = 2) -> list[int]:
    """Delete all but the newest `keep` committed checkpoints, plus any uncommitted (crashed) step dirs older than the
    newest committed one. Returns deleted steps."""
    root = Path(root)
    steps = committed_steps(root)
    newest = steps[-1] if steps else -1
    gone = []
    for p in sorted(root.iterdir()) if root.exists() else []:
        m = STEP_RE.search(p.name)
        if not m:
            continue
        s = int(m[1])
        if (s in steps[:-keep] if keep else s in steps) or (s not in steps and s < newest):
            shutil.rmtree(p)
            gone.append(s)
    return gone


def save(state: dict, root: str | Path, step: int, rank: int) -> None:
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    d = step_dir(root, step)
    dcp.save(state, checkpoint_id=str(d))
    if dist.is_initialized():
        dist.barrier()                    # every rank's shards are on disk before the marker
    if rank == 0:
        commit(root, step)


def load(state: dict, root: str | Path, step: int) -> None:
    import torch.distributed.checkpoint as dcp
    dcp.load(state, checkpoint_id=str(step_dir(root, step)))   # in place, resharding to the current layout
