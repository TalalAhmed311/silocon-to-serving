"""llama_numpy.py — the reference Llama-architecture forward pass every C++/CUDA implementation is tested against.

Matches Hugging Face `LlamaForCausalLM` semantics (checked in P1.1 example 02): RMSNorm, RoPE in the
"rotate_half" (non-interleaved) convention, grouped-query attention, SwiGLU MLP, optional tied embeddings.

Weights use HF tensor names, fp32, row-major [out_features, in_features] — the layout written by
tools/make_tiny_llama.py and tools/convert_hf_model.py.

Run:      uv run python platform/engine/v0/reference/llama_numpy.py --model build/tiny-llama --steps 16
Hardware: T0.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


@dataclass
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @classmethod
    def from_hf(cls, d: dict) -> "Config":
        return cls(dim=d["hidden_size"], hidden_dim=d["intermediate_size"], n_layers=d["num_hidden_layers"],
                   n_heads=d["num_attention_heads"], n_kv_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
                   vocab_size=d["vocab_size"], max_seq_len=d.get("max_position_embeddings", 2048),
                   rope_theta=float(d.get("rope_theta", 10000.0)), norm_eps=float(d.get("rms_norm_eps", 1e-5)),
                   tie_embeddings=bool(d.get("tie_word_embeddings", False)))


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    # Normalize by the root-mean-square (no mean subtraction, unlike LayerNorm), then scale per channel.
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def rope_tables(head_dim: int, max_pos: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))   # [hd/2]
    ang = np.outer(np.arange(max_pos, dtype=np.float64), inv_freq)                          # [pos, hd/2]
    ang = np.concatenate([ang, ang], axis=-1)                                               # HF duplicates halves
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: [..., head_dim] at one position; rotate_half pairs element i with element i + head_dim/2."""
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * cos + rot * sin


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)       # subtract the max: exp never overflows
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


class LlamaNumpy:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.cfg = Config.from_hf(json.loads((model_dir / "config.json").read_text()))
        self.w = load_file(str(model_dir / "model.safetensors"))
        c = self.cfg
        self.cos, self.sin = rope_tables(c.head_dim, c.max_seq_len, c.rope_theta)
        kv_dim = c.n_kv_heads * c.head_dim
        self.k_cache = np.zeros((c.n_layers, c.max_seq_len, kv_dim), np.float32)
        self.v_cache = np.zeros_like(self.k_cache)

    def W(self, name: str) -> np.ndarray:
        return self.w[name]

    def forward(self, token: int, pos: int) -> np.ndarray:
        """One decode step: token at position pos -> logits [vocab]. Appends K/V for pos to the cache."""
        c, hd = self.cfg, self.cfg.head_dim
        x = self.W("model.embed_tokens.weight")[token].copy()
        group = c.n_heads // c.n_kv_heads                       # query heads per KV head (GQA)
        for l in range(c.n_layers):
            p = f"model.layers.{l}."
            h = rmsnorm(x, self.W(p + "input_layernorm.weight"), c.norm_eps)
            q = (self.W(p + "self_attn.q_proj.weight") @ h).reshape(c.n_heads, hd)
            k = (self.W(p + "self_attn.k_proj.weight") @ h).reshape(c.n_kv_heads, hd)
            v = self.W(p + "self_attn.v_proj.weight") @ h
            q = apply_rope(q, self.cos[pos], self.sin[pos])
            k = apply_rope(k, self.cos[pos], self.sin[pos])
            self.k_cache[l, pos] = k.reshape(-1)
            self.v_cache[l, pos] = v
            K = self.k_cache[l, : pos + 1].reshape(pos + 1, c.n_kv_heads, hd)
            V = self.v_cache[l, : pos + 1].reshape(pos + 1, c.n_kv_heads, hd)
            out = np.empty((c.n_heads, hd), np.float32)
            for hh in range(c.n_heads):
                kvh = hh // group
                scores = (K[:, kvh, :] @ q[hh]) / np.sqrt(hd)  # causal: only positions <= pos exist
                out[hh] = softmax(scores) @ V[:, kvh, :]
            x = x + self.W(p + "self_attn.o_proj.weight") @ out.reshape(-1)
            h = rmsnorm(x, self.W(p + "post_attention_layernorm.weight"), c.norm_eps)
            g = self.W(p + "mlp.gate_proj.weight") @ h
            u = self.W(p + "mlp.up_proj.weight") @ h
            x = x + self.W(p + "mlp.down_proj.weight") @ (silu(g) * u)
        x = rmsnorm(x, self.W("model.norm.weight"), c.norm_eps)
        head = self.W("model.embed_tokens.weight") if c.tie_embeddings else self.W("lm_head.weight")
        return head @ x

    def generate_greedy(self, prompt: list[int], steps: int) -> tuple[list[int], list[np.ndarray]]:
        """Feeds the prompt, then greedy-decodes `steps` tokens. Returns (all tokens, logits per position)."""
        toks, logits_all = list(prompt), []
        for pos in range(len(prompt) + steps - 1):
            logits = self.forward(toks[pos], pos)
            logits_all.append(logits)
            if pos >= len(prompt) - 1:
                toks.append(int(np.argmax(logits)))
        return toks, logits_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="build/tiny-llama")
    ap.add_argument("--prompt", default="1 2 3 4")
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()
    m = LlamaNumpy(a.model)
    toks, _ = m.generate_greedy([int(t) for t in a.prompt.split()], a.steps)
    print(" ".join(map(str, toks)))
