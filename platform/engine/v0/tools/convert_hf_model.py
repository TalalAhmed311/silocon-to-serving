"""convert_hf_model.py — convert a Hugging Face Llama-architecture checkpoint to the engine's format (fp32 safetensors).

Run:      uv run --extra torch python platform/engine/v0/tools/convert_hf_model.py <hf_model_dir_or_id> build/my-model
Needs:    the [torch] extra; network access to the Hub if you pass a model id (or a local snapshot directory).
Good small models to try (check each license yourself): any LlamaForCausalLM ≤ ~500M params, e.g. a 135M–360M
          "SmolLM"-class model, so fp32 weights fit comfortably in laptop RAM.
What it does: loads with transformers, casts every tensor to fp32, writes model.safetensors (single file)
          + config.json + the tokenizer files (so tools/chat.py can encode/decode text).
Hardware: T0 (CPU).
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

src, dst = sys.argv[1], Path(sys.argv[2])
cfg = AutoConfig.from_pretrained(src)
if cfg.model_type != "llama":
    sys.exit(f"model_type={cfg.model_type}: the v0 engine implements the Llama architecture only")
if getattr(cfg, "rope_scaling", None):
    print(f"WARNING: rope_scaling={cfg.rope_scaling} is not implemented in the v0 engine; long-context outputs will differ")
model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
sd = {k: v.contiguous() for k, v in model.state_dict().items() if "rotary_emb" not in k}
if cfg.tie_word_embeddings:
    sd.pop("lm_head.weight", None)  # the engine reads embed_tokens for the head when tied
dst.mkdir(parents=True, exist_ok=True)
save_file(sd, str(dst / "model.safetensors"))
(dst / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
print(f"wrote {dst}: {sum(v.numel() for v in sd.values()):,} params")
