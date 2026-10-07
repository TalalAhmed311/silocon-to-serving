"""quantize.py — produce FP8 / AWQ / GPTQ checkpoints of a base model with LLM Compressor (one-shot, data-free or
calibrated). T2: calibration of a 7-8B model needs a GPU.

Run:  python platform/bakeoff/quantize.py --base /opt/models/base --scheme fp8-dynamic --out /opt/models/base-fp8
      python platform/bakeoff/quantize.py --base /opt/models/base --scheme awq-w4a16 --out /opt/models/base-awq
      python platform/bakeoff/quantize.py --base /opt/models/base --scheme gptq-w4a16 --out /opt/models/base-gptq
API:  llmcompressor 0.14.0 — `oneshot(model=..., recipe=..., dataset=..., ...)` and modifier classes. The imports below
      follow the project's README/examples at the pinned tag as the author understands them: UNVERIFIED; if an import
      fails, check `llm-compressor/examples/` at tag 0.14.0 and adjust (the scheme names are the stable part).
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--scheme", required=True, choices=["fp8-dynamic", "awq-w4a16", "gptq-w4a16"])
ap.add_argument("--out", required=True)
ap.add_argument("--calib", default="open_platypus", help="calibration dataset name (AWQ/GPTQ)")
ap.add_argument("--samples", type=int, default=256)
a = ap.parse_args()

from llmcompressor import oneshot  # noqa: E402

if a.scheme == "fp8-dynamic":
    from llmcompressor.modifiers.quantization import QuantizationModifier
    # FP8 weights (per-channel) + dynamic per-token FP8 activations: no calibration data needed. Keep lm_head in bf16.
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=a.base, recipe=recipe, output_dir=a.out)
elif a.scheme == "awq-w4a16":
    from llmcompressor.modifiers.awq import AWQModifier
    recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(model=a.base, recipe=recipe, dataset=a.calib, num_calibration_samples=a.samples, max_seq_length=2048, output_dir=a.out)
else:
    from llmcompressor.modifiers.quantization import GPTQModifier
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(model=a.base, recipe=recipe, dataset=a.calib, num_calibration_samples=a.samples, max_seq_length=2048, output_dir=a.out)
print("wrote", a.out)
