"""core.py — the capacity model (D3). Pure arithmetic: no GPU, no weights, every number traceable to an input.

Inputs:  a model (HF config.json fields), a GPU (from P1.4's gpu_specs.yaml), dtypes, TP degree, context length,
         and two explicitly-named efficiency assumptions (bandwidth and compute utilization).
Outputs: a Plan with weights/GPU, KV bytes/token, max concurrent sequences, decode ceilings per batch, prefill time.

Every formula is derived in P1.1 (params), P1.2 (KV, intensity) and P1.4 (roofline). Assumptions are parameters with
defaults, never hidden constants, and the CLI prints them next to the result.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline"))
import specs  # noqa: E402  (gpu_specs.yaml loader)

DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}


@dataclass
class Model:
    name: str
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    tie_embeddings: bool = False
    num_experts: int = 0            # MoE: total experts per layer (0 = dense)
    experts_per_token: int = 0      # MoE: active experts per token
    head_dim: int | None = None     # some configs set it explicitly (≠ hidden/heads)

    @property
    def hd(self) -> int:
        return self.head_dim or self.hidden // self.heads

    @classmethod
    def from_config(cls, cfg: dict, name: str = "model") -> "Model":
        return cls(name=name, hidden=cfg["hidden_size"], intermediate=cfg["intermediate_size"],
                   layers=cfg["num_hidden_layers"], heads=cfg["num_attention_heads"],
                   kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]), vocab=cfg["vocab_size"],
                   tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
                   num_experts=int(cfg.get("num_local_experts", cfg.get("num_experts", 0)) or 0),
                   experts_per_token=int(cfg.get("num_experts_per_tok", 0) or 0), head_dim=cfg.get("head_dim"))

    @classmethod
    def from_file(cls, path: str | Path) -> "Model":
        p = Path(path)
        return cls.from_config(json.loads(p.read_text()), name=p.parent.name or p.stem)

    # ---- parameter counts (P1.1 §2) ----
    def attn_params_per_layer(self) -> int:
        d, h = self.hidden, self.hd
        return d * self.heads * h * 2 + d * self.kv_heads * h * 2   # q, o + k, v

    def mlp_params_per_expert(self) -> int:
        return 3 * self.hidden * self.intermediate                  # gate, up, down

    def total_params(self) -> int:
        experts = max(1, self.num_experts)
        router = self.hidden * self.num_experts if self.num_experts else 0
        per_layer = self.attn_params_per_layer() + experts * self.mlp_params_per_expert() + router + 2 * self.hidden
        head = 0 if self.tie_embeddings else self.vocab * self.hidden
        return self.layers * per_layer + self.vocab * self.hidden + head + self.hidden

    def active_params(self) -> int:
        """Params whose weights are *read* per decode step: attention, the routed experts' MLPs (all of the MLP for a
        dense model), the router, norms, and the LM head. The embedding is a single-row lookup, so it is excluded —
        but the LM head is a full V×d matvec, so it is included even when tied to the embedding."""
        active = self.experts_per_token if self.num_experts else 1
        router = self.hidden * self.num_experts if self.num_experts else 0
        per_layer = self.attn_params_per_layer() + active * self.mlp_params_per_expert() + router + 2 * self.hidden
        return self.layers * per_layer + self.vocab * self.hidden + self.hidden

    def kv_bytes_per_token(self, kv_bytes: float) -> float:
        return 2 * self.layers * self.kv_heads * self.hd * kv_bytes  # P1.2


@dataclass
class Plan:
    model: str
    gpu: str
    gpu_status: str
    tp: int
    weight_dtype: str
    kv_dtype: str
    context: int
    weights_gb_per_gpu: float
    kv_budget_gb_per_gpu: float
    kv_bytes_per_token: float
    max_kv_tokens: int
    max_seqs_at_context: int
    decode_ceiling: dict = field(default_factory=dict)   # batch -> (per-seq tok/s, aggregate tok/s)
    prefill_s: dict = field(default_factory=dict)        # prompt tokens -> seconds
    fits: bool = True
    assumptions: dict = field(default_factory=dict)


def plan(model: Model, gpu_name: str, weight_dtype: str = "bf16", kv_dtype: str = "bf16", tp: int = 1,
         context: int = 4096, mem_util: float = 0.90, activation_gb: float = 1.0, bw_util: float = 0.80,
         mfu: float = 0.50, batches=(1, 8, 32, 128), prompts=(512, 2048, 8192)) -> Plan:
    g = specs.get(gpu_name)
    wb, kb = DTYPE_BYTES[weight_dtype], DTYPE_BYTES[kv_dtype]
    weights = model.total_params() * wb / tp                         # TP shards every weight matrix (≈)
    usable = g["memory_gb"] * 2**30 * mem_util                       # like vLLM's --gpu-memory-utilization
    kv_budget = usable - weights - activation_gb * 2**30
    kv_tok = model.kv_bytes_per_token(kb) / tp                       # TP shards KV heads across GPUs
    max_tokens = int(kv_budget // kv_tok) if kv_budget > 0 else 0
    p = Plan(model=model.name, gpu=g["name"], gpu_status=g["status"], tp=tp, weight_dtype=weight_dtype,
             kv_dtype=kv_dtype, context=context, weights_gb_per_gpu=weights / 2**30,
             kv_budget_gb_per_gpu=max(0.0, kv_budget) / 2**30, kv_bytes_per_token=model.kv_bytes_per_token(kb),
             max_kv_tokens=max_tokens, max_seqs_at_context=max_tokens // context, fits=kv_budget > 0,
             assumptions={"mem_util": mem_util, "activation_gb": activation_gb, "bw_util": bw_util, "mfu": mfu,
                          "note": "decode ceiling = bw_util × bandwidth ÷ bytes read per step; prefill = FLOPs ÷ (mfu × peak)"})
    bw = g["hbm_gbs"] * 1e9 * bw_util * tp                          # aggregate across TP ranks (ignores comm: P4.2)
    read_w = model.active_params() * wb                              # bytes of weights read per decode step
    for B in batches:
        if B > p.max_seqs_at_context and p.max_seqs_at_context > 0:
            continue
        # half-full context on average for a running batch: each sequence reads context/2 tokens of KV per step
        step_bytes = read_w + B * (context / 2) * model.kv_bytes_per_token(kb)
        step_s = step_bytes / bw
        p.decode_ceiling[B] = (1 / step_s, B / step_s)
    peak = g.get(f"{'fp8' if weight_dtype == 'fp8' else 'fp16'}_dense_tflops")
    if peak:
        for T in prompts:
            flops = 2 * model.active_params() * T + 2 * model.layers * model.heads * model.hd * T * T  # weights + causal attn
            p.prefill_s[T] = flops / (peak * 1e12 * mfu * tp)
    return p


def presets_dir() -> Path:
    return Path(__file__).resolve().parent / "presets"
