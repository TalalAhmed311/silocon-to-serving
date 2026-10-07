"""model_runner.py — turns a SchedulerOutput into logits (P6.1/P6.6). Three runners share one interface:

  runner.execute(out: SchedulerOutput) -> dict[seq_id, logits row]   # only for chunks with produces_token

  FakeRunner        T0. Logits are a pure function of the token history (a hash), so outputs must be identical
                    however requests are batched, chunked, preempted or prefix-cached — the engine's core test.
  NumpyPagedRunner  T0. The real Llama forward (same math as v0 reference/llama_numpy.py) over a PAGED KV pool:
                    K/V live at slot = block_table[pos // bs]·bs + pos % bs. Must match v0's greedy output exactly.
  TorchRunner       T2 (one GPU). Batched varlen prefill + batched decode with a gather-based paged attention whose
                    shapes are static per batch size, so decode can be captured in CUDA graphs (cuda_graph.py).
                    TODO(run-on: T2) — never executed in this repo yet.
Every runner executes out.copy_ops (copy-on-write block copies) before computing anything.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .scheduler import SchedulerOutput

_V0_REF = Path(__file__).resolve().parents[2] / "v0" / "reference"


# ---- T0: fake model ----------------------------------------------------------------------------------------------
class FakeRunner:
    def __init__(self, vocab_size: int = 101, eos_id: int | None = None):
        self.vocab_size, self.eos_id = vocab_size, eos_id
        self.calls: list[int] = []                      # tokens per forward (tests check the budget held)

    def logits_for(self, tokens: list[int]) -> np.ndarray:
        h = hashlib.sha256(",".join(map(str, tokens)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return rng.standard_normal(self.vocab_size).astype(np.float32)

    def execute(self, out: SchedulerOutput) -> dict[int, np.ndarray]:
        self.calls.append(out.num_tokens)
        res = {}
        for c in out.chunks:
            if c.produces_token:
                res[c.seq.seq_id] = self.logits_for(c.seq.tokens[: c.start + c.num_tokens])
        return res


# ---- T0: NumPy Llama over a paged KV pool -----------------------------------------------------------------------
class NumpyPagedRunner:
    def __init__(self, model_dir: str | Path, num_blocks: int, block_size: int):
        sys.path.insert(0, str(_V0_REF))
        import llama_numpy as ln                         # the v0 reference: same weights, same math
        self.ln = ln
        model_dir = Path(model_dir)
        self.cfg = ln.Config.from_hf(json.loads((model_dir / "config.json").read_text()))
        from safetensors.numpy import load_file
        self.w = load_file(str(model_dir / "model.safetensors"))
        c = self.cfg
        self.cos, self.sin = ln.rope_tables(c.head_dim, c.max_seq_len, c.rope_theta)
        self.block_size = block_size
        kv_dim = c.n_kv_heads * c.head_dim
        self.k = np.zeros((c.n_layers, num_blocks * block_size, kv_dim), np.float32)   # the paged pool
        self.v = np.zeros_like(self.k)

    def copy_blocks(self, ops) -> None:
        bs = self.block_size
        for src, dst in ops:
            self.k[:, dst * bs:(dst + 1) * bs] = self.k[:, src * bs:(src + 1) * bs]
            self.v[:, dst * bs:(dst + 1) * bs] = self.v[:, src * bs:(src + 1) * bs]

    def _slot(self, table, pos):
        return table[pos // self.block_size] * self.block_size + pos % self.block_size

    def _forward_one(self, table, token: int, pos: int) -> np.ndarray:
        """llama_numpy.forward with the contiguous cache replaced by slot lookups."""
        ln, c, W = self.ln, self.cfg, self.w
        hd, group = c.head_dim, c.n_heads // c.n_kv_heads
        x = W["model.embed_tokens.weight"][token].copy()
        slots = [self._slot(table, p) for p in range(pos + 1)]
        for l in range(c.n_layers):
            p = f"model.layers.{l}."
            h = ln.rmsnorm(x, W[p + "input_layernorm.weight"], c.norm_eps)
            q = (W[p + "self_attn.q_proj.weight"] @ h).reshape(c.n_heads, hd)
            k = (W[p + "self_attn.k_proj.weight"] @ h).reshape(c.n_kv_heads, hd)
            v = W[p + "self_attn.v_proj.weight"] @ h
            q = ln.apply_rope(q, self.cos[pos], self.sin[pos])
            k = ln.apply_rope(k, self.cos[pos], self.sin[pos])
            self.k[l, slots[-1]] = k.reshape(-1)
            self.v[l, slots[-1]] = v
            K = self.k[l, slots].reshape(pos + 1, c.n_kv_heads, hd)       # gather: the paged-attention read
            V = self.v[l, slots].reshape(pos + 1, c.n_kv_heads, hd)
            out = np.empty((c.n_heads, hd), np.float32)
            for hh in range(c.n_heads):
                kvh = hh // group
                out[hh] = ln.softmax((K[:, kvh] @ q[hh]) / np.sqrt(hd)) @ V[:, kvh]
            x = x + W[p + "self_attn.o_proj.weight"] @ out.reshape(-1)
            h = ln.rmsnorm(x, W[p + "post_attention_layernorm.weight"], c.norm_eps)
            x = x + W[p + "mlp.down_proj.weight"] @ (ln.silu(W[p + "mlp.gate_proj.weight"] @ h) * (W[p + "mlp.up_proj.weight"] @ h))
        x = ln.rmsnorm(x, W["model.norm.weight"], c.norm_eps)
        head = W["model.embed_tokens.weight"] if c.tie_embeddings else W["lm_head.weight"]
        return head @ x

    def execute(self, out: SchedulerOutput) -> dict[int, np.ndarray]:
        self.copy_blocks(out.copy_ops)
        res = {}
        for c in out.chunks:
            toks = c.seq.tokens
            logits = None
            for pos in range(c.start, c.start + c.num_tokens):     # token-at-a-time: slow, obviously correct
                logits = self._forward_one(c.seq.block_table, toks[pos], pos)
            if c.produces_token:
                res[c.seq.seq_id] = logits
        return res


# ---- T2: PyTorch on a GPU ---------------------------------------------------------------------------------------
class TorchRunner:
    """Llama in plain PyTorch ops with a paged KV pool [layers, 2, num_blocks·bs, kv_heads, hd].
    Prefill chunks: per-sequence causal attention over gathered slots (varlen; a FlashAttention varlen kernel is the
    upgrade — exercise 5). Decode: one batched pass; attention gathers each sequence's slots through a padded
    [B, max_blocks] block table, so for fixed (B, max_blocks) every tensor shape is static → CUDA-graph capturable.
    TODO(run-on: T2): validate against NumpyPagedRunner on the tiny model (tests/test_torch_runner.py, marked gpu)."""

    def __init__(self, model_dir, num_blocks: int, block_size: int, dtype: str = "bfloat16", device: str = "cuda",
                 max_blocks_per_seq: int = 256):
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        model_dir = Path(model_dir)
        hf = json.loads((model_dir / "config.json").read_text())
        sys.path.insert(0, str(_V0_REF))
        import llama_numpy as ln
        self.cfg = c = ln.Config.from_hf(hf)
        self.dtype, self.device = getattr(torch, dtype), torch.device(device)
        self.w = {k: v.to(self.device, self.dtype) for k, v in load_file(str(model_dir / "model.safetensors")).items()}
        cos, sin = ln.rope_tables(c.head_dim, c.max_seq_len, c.rope_theta)
        self.cos = torch.tensor(cos, device=self.device, dtype=torch.float32)
        self.sin = torch.tensor(sin, device=self.device, dtype=torch.float32)
        self.block_size, self.max_blocks = block_size, max_blocks_per_seq
        self.kv = torch.zeros((c.n_layers, 2, num_blocks * block_size, c.n_kv_heads, c.head_dim),
                              device=self.device, dtype=self.dtype)
        self.graphs = None                                          # set by cuda_graph.capture_decode()

    # -- small ops (fp32 math for norms/softmax, as the reference does) --
    def _rms(self, x, w):
        xf = x.float()
        return (xf * self.torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.norm_eps)).to(x.dtype) * w

    def _rope(self, x, pos):                                         # x [T, H, hd], pos [T]
        h = x.shape[-1] // 2
        cos, sin = self.cos[pos][:, None, :], self.sin[pos][:, None, :]
        xf = x.float()
        rot = self.torch.cat([-xf[..., h:], xf[..., :h]], dim=-1)
        return (xf * cos + rot * sin).to(x.dtype)

    def _slots(self, block_table, positions):                       # block_table [.., max_blocks] long, positions [..]
        return block_table.gather(-1, (positions // self.block_size).unsqueeze(-1)).squeeze(-1) * self.block_size \
            + positions % self.block_size

    def copy_blocks(self, ops) -> None:
        bs = self.block_size
        for src, dst in ops:
            self.kv[:, :, dst * bs:(dst + 1) * bs] = self.kv[:, :, src * bs:(src + 1) * bs]

    # -- one transformer pass over T flat tokens; attn_fn decides how attention reads the cache --
    def _layers(self, tokens, positions, write_slots, attn_fn):
        torch, c, W = self.torch, self.cfg, self.w
        hd = c.head_dim
        x = W["model.embed_tokens.weight"][tokens]
        for l in range(c.n_layers):
            p = f"model.layers.{l}."
            h = self._rms(x, W[p + "input_layernorm.weight"])
            q = (h @ W[p + "self_attn.q_proj.weight"].T).view(-1, c.n_heads, hd)
            k = (h @ W[p + "self_attn.k_proj.weight"].T).view(-1, c.n_kv_heads, hd)
            v = (h @ W[p + "self_attn.v_proj.weight"].T).view(-1, c.n_kv_heads, hd)
            q, k = self._rope(q, positions), self._rope(k, positions)
            self.kv[l, 0].index_copy_(0, write_slots, k)               # write before read: a token attends to itself
            self.kv[l, 1].index_copy_(0, write_slots, v)
            o = attn_fn(l, q)                                          # [T, H, hd]
            x = x + o.reshape(o.shape[0], -1) @ W[p + "self_attn.o_proj.weight"].T
            h = self._rms(x, W[p + "post_attention_layernorm.weight"])
            g = h @ W[p + "mlp.gate_proj.weight"].T
            u = h @ W[p + "mlp.up_proj.weight"].T
            x = x + (torch.nn.functional.silu(g) * u) @ W[p + "mlp.down_proj.weight"].T
        return self._rms(x, W["model.norm.weight"])

    def _head(self, x):
        W = self.w
        head = W["model.embed_tokens.weight"] if self.cfg.tie_embeddings else W["lm_head.weight"]
        return (x @ head.T).float()

    def _attend(self, q, K, V, mask):
        """q [B, Tq, H, hd]; K, V [B, S, KVH, hd]; mask [B, Tq, S] bool (True = may attend). GQA by head repeat."""
        torch, c = self.torch, self.cfg
        group = c.n_heads // c.n_kv_heads
        K = K.repeat_interleave(group, dim=2)
        V = V.repeat_interleave(group, dim=2)
        s = torch.einsum("bqhd,bshd->bhqs", q.float(), K.float()) / c.head_dim ** 0.5
        s = s.masked_fill(~mask[:, None], float("-inf"))
        return torch.einsum("bhqs,bshd->bqhd", torch.softmax(s, -1), V.float()).to(q.dtype)

    def prefill_chunk(self, seq, start: int, n: int):
        torch = self.torch
        toks = torch.tensor(seq.tokens[start:start + n], device=self.device)
        pos = torch.arange(start, start + n, device=self.device)
        table = torch.tensor(seq.block_table, device=self.device, dtype=torch.long)
        write = self._slots(table[None].expand(n, -1), pos)
        all_pos = torch.arange(start + n, device=self.device)
        read = self._slots(table[None].expand(start + n, -1), all_pos)
        mask = (all_pos[None, :] <= pos[:, None])[None]                       # causal, incl. cached prefix

        def attn(l, q):
            K, V = self.kv[l, 0][read][None], self.kv[l, 1][read][None]
            return self._attend(q[None], K, V, mask)[0]
        return self._layers(toks, pos, write, attn)[-1:]                       # last position's hidden state

    def decode(self, tokens, positions, block_tables):
        """tokens, positions [B]; block_tables [B, max_blocks] (padded with 0). Static shapes for fixed B."""
        torch = self.torch
        S = block_tables.shape[1] * self.block_size
        all_pos = torch.arange(S, device=self.device)
        read = self._slots(block_tables[:, None, :].expand(-1, S, -1), all_pos[None].expand(block_tables.shape[0], -1))
        mask = (all_pos[None, :] <= positions[:, None])[:, None, :]             # [B, 1, S]
        write = self._slots(block_tables, positions)

        def attn(l, q):
            K, V = self.kv[l, 0][read], self.kv[l, 1][read]                    # [B, S, KVH, hd] gather
            return self._attend(q[:, None], K, V, mask)[:, 0]
        return self._head(self._layers(tokens, positions, write, attn))

    def execute(self, out: SchedulerOutput) -> dict[int, "object"]:
        torch = self.torch
        self.copy_blocks(out.copy_ops)
        res = {}
        decodes = [c for c in out.chunks if c.num_tokens == 1 and not c.seq.in_prefill]
        decode_ids = {id(c) for c in decodes}
        for c in out.chunks:
            if id(c) in decode_ids:
                continue
            h = self.prefill_chunk(c.seq, c.start, c.num_tokens)
            if c.produces_token:
                res[c.seq.seq_id] = self._head(h)[0]
        if decodes:
            B = len(decodes)
            tok = torch.tensor([c.seq.tokens[c.start] for c in decodes], device=self.device)
            pos = torch.tensor([c.start for c in decodes], device=self.device)
            tables = torch.zeros((B, self.max_blocks), dtype=torch.long, device=self.device)
            for i, c in enumerate(decodes):
                tables[i, :len(c.seq.block_table)] = torch.tensor(c.seq.block_table, device=self.device)
            logits = self.graphs.run(tok, pos, tables) if self.graphs else self.decode(tok, pos, tables)
            for i, c in enumerate(decodes):
                res[c.seq.seq_id] = logits[i]
        return res
