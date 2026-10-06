"""01_numpy_llama_counted.py — the P0.5 NumPy Llama, instrumented to count FLOPs and bytes per op for one decode step.

Run:      uv run python course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/01_numpy_llama_counted.py
          [--model DIR] [--pos 100] [--bytes-per-param 2]
Expected: a table of ops with FLOPs and weight bytes; matmuls dominate. Counts are exact (no timing involved).
Hardware: T0. Uses a tiny random model unless --model points at a converted HF model directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
V0 = ROOT / "platform" / "engine" / "v0"
spec = importlib.util.spec_from_file_location("llama_numpy", V0 / "reference" / "llama_numpy.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class Counted(ref.LlamaNumpy):
    """Same math as the reference; every matmul and elementwise op is tallied."""

    def __init__(self, model_dir, bytes_per_param: float):
        super().__init__(model_dir)
        self.bpp = bytes_per_param
        self.flops: dict[str, float] = defaultdict(float)
        self.wbytes: dict[str, float] = defaultdict(float)

    def mm(self, name: str, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        self.flops[name] += 2 * W.size          # one multiply + one add per weight
        self.wbytes[name] += W.size * self.bpp  # every weight read once
        return W @ x

    def forward(self, token: int, pos: int) -> np.ndarray:
        c, hd = self.cfg, self.cfg.head_dim
        x = self.W("model.embed_tokens.weight")[token].copy()
        self.wbytes["embedding (1 row)"] += c.dim * self.bpp
        group = c.n_heads // c.n_kv_heads
        for l in range(c.n_layers):
            p = f"model.layers.{l}."
            h = ref.rmsnorm(x, self.W(p + "input_layernorm.weight"), c.norm_eps)
            self.flops["rmsnorm"] += 4 * c.dim
            q = self.mm("q_proj", self.W(p + "self_attn.q_proj.weight"), h).reshape(c.n_heads, hd)
            k = self.mm("k_proj", self.W(p + "self_attn.k_proj.weight"), h).reshape(c.n_kv_heads, hd)
            v = self.mm("v_proj", self.W(p + "self_attn.v_proj.weight"), h)
            q = ref.apply_rope(q, self.cos[pos], self.sin[pos])
            k = ref.apply_rope(k, self.cos[pos], self.sin[pos])
            self.flops["rope"] += 3 * (q.size + k.size)
            self.k_cache[l, pos] = k.reshape(-1)
            self.v_cache[l, pos] = v
            K = self.k_cache[l, : pos + 1].reshape(pos + 1, c.n_kv_heads, hd)
            V = self.v_cache[l, : pos + 1].reshape(pos + 1, c.n_kv_heads, hd)
            out = np.empty((c.n_heads, hd), np.float32)
            for hh in range(c.n_heads):
                s = (K[:, hh // group, :] @ q[hh]) / np.sqrt(hd)
                out[hh] = ref.softmax(s) @ V[:, hh // group, :]
            t = pos + 1
            self.flops["attention q·K and p·V"] += 4 * c.n_heads * hd * t
            self.wbytes["KV cache read"] += 2 * t * c.n_kv_heads * hd * self.bpp
            x = x + self.mm("o_proj", self.W(p + "self_attn.o_proj.weight"), out.reshape(-1))
            h = ref.rmsnorm(x, self.W(p + "post_attention_layernorm.weight"), c.norm_eps)
            self.flops["rmsnorm"] += 4 * c.dim
            g = self.mm("gate_proj", self.W(p + "mlp.gate_proj.weight"), h)
            u = self.mm("up_proj", self.W(p + "mlp.up_proj.weight"), h)
            self.flops["silu·mul"] += 5 * g.size
            x = x + self.mm("down_proj", self.W(p + "mlp.down_proj.weight"), ref.silu(g) * u)
        x = ref.rmsnorm(x, self.W("model.norm.weight"), c.norm_eps)
        head = self.W("model.embed_tokens.weight") if c.tie_embeddings else self.W("lm_head.weight")
        return self.mm("lm_head", head, x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--bytes-per-param", type=float, default=2.0, help="2 = fp16/bf16 storage")
    a = ap.parse_args()
    model = a.model
    if model is None:
        model = tempfile.mkdtemp()
        subprocess.run([sys.executable, str(V0 / "tools" / "make_tiny_llama.py"), model, "--max-seq", str(a.pos + 2)],
                       check=True, capture_output=True)
    m = Counted(model, a.bytes_per_param)
    for p in range(a.pos):            # fill the cache; reset the counters; count only the step at `pos`
        m.forward(1, p)
    m.flops.clear(); m.wbytes.clear()
    m.forward(1, a.pos)
    tf, tb = sum(m.flops.values()), sum(m.wbytes.values())
    print(f"one decode step at position {a.pos}: {tf:,.0f} FLOPs, {tb:,.0f} bytes read -> intensity {tf / tb:.2f} FLOP/B\n")
    print("| op | FLOPs | % | bytes read | % |\n|---|---|---|---|---|")
    for k in sorted(set(m.flops) | set(m.wbytes), key=lambda k: -m.flops.get(k, 0)):
        f, b = m.flops.get(k, 0), m.wbytes.get(k, 0)
        print(f"| {k} | {f:,.0f} | {100 * f / tf:.1f} | {b:,.0f} | {100 * b / tb:.1f} |")


if __name__ == "__main__":
    main()
