"""02_hf_parity.py — prove the NumPy reference equals Hugging Face LlamaForCausalLM, logit for logit.

Run:      uv run --extra torch python course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/02_hf_parity.py
Expected: "max |Δlogit| = <1e-5-ish>" and "PARITY OK" (threshold 1e-4, fp32 on CPU).
Hardware: T0 (CPU). No network: the model is random-init from a LlamaConfig.
"""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("llama_numpy", ROOT / "platform/engine/v0/reference/llama_numpy.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

torch.manual_seed(0)
cfg = LlamaConfig(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=256, max_position_embeddings=64, rms_norm_eps=1e-5,
                  rope_theta=10000.0, tie_word_embeddings=False)
model = LlamaForCausalLM(cfg).eval().float()

d = Path(tempfile.mkdtemp())
save_file({k: v.contiguous() for k, v in model.state_dict().items() if "rotary" not in k}, str(d / "model.safetensors"))
(d / "config.json").write_text(cfg.to_json_string())

ids = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38]
with torch.no_grad():
    hf = model(torch.tensor([ids])).logits[0].numpy()        # [T, V]: one causal forward over the whole sequence
m = ref.LlamaNumpy(d)
ours = np.stack([m.forward(t, p) for p, t in enumerate(ids)])  # one token at a time with the KV cache
err = np.abs(hf - ours).max()
print(f"max |Δlogit| = {err:.2e} over {len(ids)} positions × {cfg.vocab_size} logits")
assert err < 1e-4, "PARITY FAILED — check RoPE convention, GQA head mapping, eps"
print("PARITY OK: the NumPy reference matches HF (batched causal forward == incremental decode with KV cache)")
