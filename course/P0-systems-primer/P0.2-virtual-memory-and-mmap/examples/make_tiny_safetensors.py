"""make_tiny_safetensors.py — write a small deterministic .safetensors file for the C++ loader to read.

Run:      uv run python examples/make_tiny_safetensors.py /tmp/tiny.safetensors
Expected: prints each tensor's name, dtype, shape and sum; 05_safetensors_mmap must print the same sums.
Hardware: T0. Uses the official `safetensors` writer so the file is guaranteed to follow the spec.
"""
import sys

import numpy as np
from safetensors.numpy import save_file

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny.safetensors"
rng = np.random.default_rng(0)
tensors = {
    "embed.weight": rng.standard_normal((32, 16)).astype(np.float32),
    "layers.0.attn.wq": rng.standard_normal((16, 16)).astype(np.float32),
    "layers.0.norm.weight": np.ones(16, dtype=np.float32),
    "lm_head.weight_f16": rng.standard_normal((32, 16)).astype(np.float16),
    "counts_i32": np.arange(10, dtype=np.int32),
}
save_file(tensors, path, metadata={"format": "s2s-tiny"})
for name in sorted(tensors):
    t = tensors[name]
    print(f"{name:24s} {str(t.dtype):8s} {list(t.shape)!s:10s} sum={float(t.astype(np.float64).sum()):.6f}")
