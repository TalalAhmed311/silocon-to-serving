"""chat.py — text in, text out, using a converted HF model's tokenizer around the C++ engine.

Run:      uv run --extra torch python platform/engine/v0/tools/chat.py --model build/my-model "Once upon a time"
Needs:    a model converted with convert_hf_model.py (it saves the tokenizer next to the weights),
          and the engine built at build/engine-v0/s2s-engine.
Hardware: T0.
"""
import argparse
import subprocess

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--model", required=True)
ap.add_argument("--engine", default="build/engine-v0/s2s-engine")
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--top-p", type=float, default=0.9)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
ids = tok(a.prompt)["input_ids"]
out = subprocess.run([a.engine, "--model", a.model, "--prompt-ids", " ".join(map(str, ids)), "--steps", str(a.steps),
                      "--temp", str(a.temp), "--top-p", str(a.top_p), "--max-seq", str(len(ids) + a.steps + 1)],
                     capture_output=True, text=True, check=True)
print(tok.decode([int(t) for t in out.stdout.split()]))
print(out.stderr.strip())
