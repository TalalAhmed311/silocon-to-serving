"""03_tiny_hf_spec.py — #10 end to end on CPU with two tiny random-init Llamas (draft = 1 layer, target = 2 layers).

Run: uv run --extra torch python course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/03_tiny_hf_spec.py
Expected: low α (random models disagree), tokens/target call between 1 and k+1, and greedy spec == greedy target-only.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform"))
from specdec.core import generate  # noqa: E402
from specdec.hf_models import HFModel  # noqa: E402

torch.manual_seed(0)
mk = lambda layers: LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=layers,  # noqa: E731
                                                 num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                                                 max_position_embeddings=256))
target, draft = HFModel(mk(2)), HFModel(mk(1))
out, st = generate([1, 5, 9], draft, target, k=4, n_tokens=48, seed=0)
print(f"α = {st.acceptance_rate:.2f}, tokens/target call = {st.tokens_per_target_call:.2f}, rounds = {st.rounds}")

# Greedy check: with temperature → 0 both rules reduce to argmax agreement; outputs must be identical.
g_target, g_draft = HFModel(target.m, temperature=1e-4), HFModel(draft.m, temperature=1e-4)
spec, _ = generate([1, 5, 9], g_draft, g_target, k=4, n_tokens=32, seed=0)
ref = [1, 5, 9]
for _ in range(32):
    ref.append(int(np.argmax(g_target.probs(ref))))
print("greedy spec == greedy target-only:", spec == ref)
