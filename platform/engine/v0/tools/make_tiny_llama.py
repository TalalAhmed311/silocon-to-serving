"""make_tiny_llama.py — write a deterministic random-init tiny Llama (HF tensor names, fp32) for tests and CI.

Run:      uv run python platform/engine/v0/tools/make_tiny_llama.py build/tiny-llama [--dim 64 --layers 2 ...]
Output:   <dir>/config.json (HF LlamaConfig keys) and <dir>/model.safetensors.
Why random weights: the course can't assume network access to a model hub in CI; correctness tests only need
the C++ engine and the NumPy reference to agree on the same weights. For real text, see convert_hf_model.py.
Hardware: T0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--hidden", type=int, default=160)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--kv-heads", type=int, default=2)
ap.add_argument("--vocab", type=int, default=256)
ap.add_argument("--max-seq", type=int, default=128)
ap.add_argument("--tie", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
hd = a.dim // a.heads


def mat(out: int, inp: int) -> np.ndarray:
    # std 1/sqrt(in): keeps activations O(1) through the layers so logits are well separated (fewer argmax ties).
    return (rng.standard_normal((out, inp)) / np.sqrt(inp)).astype(np.float32)


w = {"model.embed_tokens.weight": rng.standard_normal((a.vocab, a.dim)).astype(np.float32)}
for l in range(a.layers):
    p = f"model.layers.{l}."
    w[p + "input_layernorm.weight"] = (1 + 0.1 * rng.standard_normal(a.dim)).astype(np.float32)
    w[p + "self_attn.q_proj.weight"] = mat(a.heads * hd, a.dim)
    w[p + "self_attn.k_proj.weight"] = mat(a.kv_heads * hd, a.dim)
    w[p + "self_attn.v_proj.weight"] = mat(a.kv_heads * hd, a.dim)
    w[p + "self_attn.o_proj.weight"] = mat(a.dim, a.heads * hd)
    w[p + "post_attention_layernorm.weight"] = (1 + 0.1 * rng.standard_normal(a.dim)).astype(np.float32)
    w[p + "mlp.gate_proj.weight"] = mat(a.hidden, a.dim)
    w[p + "mlp.up_proj.weight"] = mat(a.hidden, a.dim)
    w[p + "mlp.down_proj.weight"] = mat(a.dim, a.hidden)
w["model.norm.weight"] = np.ones(a.dim, np.float32)
if not a.tie:
    w["lm_head.weight"] = mat(a.vocab, a.dim) * 4  # sharper logits

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
save_file(w, str(out / "model.safetensors"))
cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": a.dim,
       "intermediate_size": a.hidden, "num_hidden_layers": a.layers, "num_attention_heads": a.heads,
       "num_key_value_heads": a.kv_heads, "vocab_size": a.vocab, "max_position_embeddings": a.max_seq,
       "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": a.tie, "hidden_act": "silu",
       "torch_dtype": "float32"}
(out / "config.json").write_text(json.dumps(cfg, indent=2))
n = sum(v.size for v in w.values())
print(f"wrote {out} ({n:,} params, {n * 4 / 2**20:.2f} MiB fp32)")
