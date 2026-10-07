"""bench.py — #5 loader benchmark: where weight-load time goes.

    PYTHONPATH=platform python -m weights.bench --dir <local model dir> [--s3 s3://bucket/prefix/model/version] [--gpu]
Prints  | method | model GB | load s | effective GB/s |  and writes results/p3.7-loader.json.

Methods:
  read-cold      sequential read of every shard after dropping the page cache (needs root: echo 3 > /proc/sys/vm/drop_caches)
  read-warm      the same, from the page cache (an upper bound for "disk" reads)
  safetensors    safetensors.numpy.load_file() of every shard (parse + copy into numpy arrays)
  mmap           safetensors safe_open(...) + get_tensor on every tensor (lazy, page-faults on access)
  s3-download    S3Store.fetch with --concurrency N into a temp dir   (T2/T3, on an instance in the bucket's region)
  to-gpu         torch: safetensors → pinned host → cuda tensors      (T2, --gpu)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .manifest import safetensors_shards


def drop_caches() -> bool:
    try:
        subprocess.run(["sync"], check=True)
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True
    except OSError:
        return False


def t_read(files):
    t = time.perf_counter()
    for f in files:
        with open(f, "rb", buffering=0) as fh:
            while fh.read(64 << 20):
                pass
    return time.perf_counter() - t


def t_safetensors(files):
    from safetensors.numpy import load_file
    t = time.perf_counter()
    for f in files:
        load_file(str(f))
    return time.perf_counter() - t


def t_mmap(files):
    import numpy as np
    from safetensors import safe_open
    t = time.perf_counter()
    for f in files:
        with safe_open(str(f), framework="numpy") as st:
            for k in st.keys():
                np.asarray(st.get_tensor(k)).sum()   # touch it, so the pages are actually read
    return time.perf_counter() - t


def t_gpu(files):
    import torch
    from safetensors.torch import load_file
    torch.cuda.synchronize()
    t = time.perf_counter()
    for f in files:
        for v in load_file(str(f), device="cpu").values():
            v.pin_memory().to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t


def t_s3(uri, concurrency):
    from .cache import S3Store
    bucket_path, _, rest = uri.removeprefix("s3://").partition("/")
    *prefix, model, version = rest.split("/")
    st = S3Store("s3://" + "/".join([bucket_path, *prefix]), max_concurrency=concurrency)
    man = st.manifest(model, version)
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        for e in man["files"]:
            st.fetch(model, version, e["path"], Path(d) / e["path"])
        return time.perf_counter() - t, man["total_bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--s3")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="results/p3.7-loader.json")
    a = ap.parse_args()
    root = Path(a.dir)
    files = [root / s for s in safetensors_shards(root)]
    gb = sum(f.stat().st_size for f in files) / 1e9
    rows = []
    if drop_caches():
        rows.append(("read-cold", gb, t_read(files)))
    else:
        print("# read-cold skipped: needs root to drop the page cache")
    t_read(files)
    rows.append(("read-warm", gb, t_read(files)))
    rows.append(("safetensors", gb, t_safetensors(files)))
    rows.append(("mmap", gb, t_mmap(files)))
    if a.gpu:
        rows.append(("to-gpu", gb, t_gpu(files)))
    if a.s3:
        for c in a.concurrency:
            s, nbytes = t_s3(a.s3, c)
            rows.append((f"s3-download c={c}", nbytes / 1e9, s))
    print("| method | model GB | load s | effective GB/s |\n|---|---|---|---|")
    for m, g, s in rows:
        print(f"| {m} | {g:.2f} | {s:.2f} | {g / s:.2f} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Path(a.out).write_text(json.dumps([{"method": m, "gb": g, "seconds": s} for m, g, s in rows], indent=2))


if __name__ == "__main__":
    main()
