"""Exercise 3 solution: real batched prefill (GEMMs + causal mask) followed by cached decode."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
spec = importlib.util.spec_from_file_location("llama_numpy", ROOT / "platform/engine/v0/reference/llama_numpy.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class PrefillThenDecode:
    def __init__(self, model_dir):
        base = ref.LlamaNumpy(model_dir)          # reuse loading only
        self.cfg, self.w, self.cos, self.sin = base.cfg, base.w, base.cos, base.sin
        c = self.cfg
        kv = c.n_kv_heads * c.head_dim
        self.K = np.zeros((c.n_layers, c.max_seq_len, kv), np.float32)
        self.V = np.zeros_like(self.K)
        self.pos = 0

    def _block(self, x: np.ndarray, start: int) -> np.ndarray:
        """x: [T, d] for positions start..start+T-1. Attends to cache[0:start] plus itself (causal)."""
        c, hd = self.cfg, self.cfg.head_dim
        T, group = x.shape[0], c.n_heads // c.n_kv_heads
        positions = np.arange(start, start + T)
        for l in range(c.n_layers):
            p = f"model.layers.{l}."
            h = ref.rmsnorm(x, self.w[p + "input_layernorm.weight"], c.norm_eps)
            q = (h @ self.w[p + "self_attn.q_proj.weight"].T).reshape(T, c.n_heads, hd)   # GEMM: [T,d]x[d,d]
            k = (h @ self.w[p + "self_attn.k_proj.weight"].T).reshape(T, c.n_kv_heads, hd)
            v = h @ self.w[p + "self_attn.v_proj.weight"].T
            q = ref.apply_rope(q, self.cos[positions][:, None, :], self.sin[positions][:, None, :])
            k = ref.apply_rope(k, self.cos[positions][:, None, :], self.sin[positions][:, None, :])
            self.K[l, start:start + T] = k.reshape(T, -1)
            self.V[l, start:start + T] = v
            Kall = self.K[l, : start + T].reshape(start + T, c.n_kv_heads, hd)
            Vall = self.V[l, : start + T].reshape(start + T, c.n_kv_heads, hd)
            # causal mask: query at absolute position start+i may see keys 0..start+i
            mask = np.arange(start + T)[None, :] > positions[:, None]
            out = np.empty((T, c.n_heads, hd), np.float32)
            for hh in range(c.n_heads):
                s = (q[:, hh, :] @ Kall[:, hh // group, :].T) / np.sqrt(hd)   # [T, start+T]
                s = np.where(mask, -np.inf, s)
                out[:, hh, :] = ref.softmax(s, axis=-1) @ Vall[:, hh // group, :]
            x = x + out.reshape(T, -1) @ self.w[p + "self_attn.o_proj.weight"].T
            h = ref.rmsnorm(x, self.w[p + "post_attention_layernorm.weight"], c.norm_eps)
            g = h @ self.w[p + "mlp.gate_proj.weight"].T
            u = h @ self.w[p + "mlp.up_proj.weight"].T
            x = x + (ref.silu(g) * u) @ self.w[p + "mlp.down_proj.weight"].T
        x = ref.rmsnorm(x, self.w["model.norm.weight"], c.norm_eps)
        head = self.w["model.embed_tokens.weight"] if c.tie_embeddings else self.w["lm_head.weight"]
        return x @ head.T

    def prefill_all_logits(self, tokens):
        self.pos = 0
        x = self.w["model.embed_tokens.weight"][np.array(tokens)].copy()
        out = self._block(x, 0)
        self.pos = len(tokens)
        return out

    def prefill(self, tokens):
        return self.prefill_all_logits(tokens)[-1]

    def decode(self, token):
        x = self.w["model.embed_tokens.weight"][[token]].copy()
        out = self._block(x, self.pos)[0]
        self.pos += 1
        return out
