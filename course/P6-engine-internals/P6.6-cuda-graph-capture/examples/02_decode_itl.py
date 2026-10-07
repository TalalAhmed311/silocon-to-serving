"""Decode ms/step, eager vs CUDA graphs, with #0 v1's TorchRunner on the tiny model (or --model DIR). T2.

Run: uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/02_decode_itl.py [--eager]
TODO(run-on: g6.xlarge)
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "platform" / "engine" / "v1"))
from s2s_engine.cuda_graph import capture_decode  # noqa: E402
from s2s_engine.model_runner import TorchRunner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--eager", action="store_true")
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()
model = a.model
if model is None:
    model = tempfile.mkdtemp()
    subprocess.run([sys.executable, str(ROOT / "platform/engine/v0/tools/make_tiny_llama.py"), model], check=True)

bs, mb = 16, 64
r = TorchRunner(model, num_blocks=2048, block_size=bs, dtype=a.dtype, max_blocks_per_seq=mb)
if not a.eager:
    capture_decode(r, batch_sizes=(1, 8, 32))
for B in (1, 8, 32):
    tok = torch.zeros(B, dtype=torch.long, device="cuda")
    pos = torch.full((B,), 100, dtype=torch.long, device="cuda")
    tables = torch.arange(B * mb, device="cuda").view(B, mb) % 2047 + 1     # distinct blocks, block 0 reserved
    fn = (lambda: r.decode(tok, pos, tables)) if a.eager else (lambda: r.graphs.run(tok, pos, tables))
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    print(f"{'eager ' if a.eager else 'graphs'} B={B:3d}: {(time.perf_counter() - t) / 50 * 1e3:.3f} ms/step")
