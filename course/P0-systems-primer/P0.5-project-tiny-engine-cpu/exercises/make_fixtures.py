"""make_fixtures.py — NumPy reference outputs for the P0.5 exercises (deterministic, seed 0).

Run:      uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/exercises/make_fixtures.py build/p05-fixtures
Writes:   <dir>/ops.safetensors      rmsnorm / rope / attention inputs and expected outputs
          <dir>/tiny/                a tiny random Llama (via platform/engine/v0/tools/make_tiny_llama.py)
          <dir>/forward.safetensors  teacher-forced logits and greedy tokens from the NumPy reference
Hardware: T0.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[4]
V0 = ROOT / "platform" / "engine" / "v0"
spec = importlib.util.spec_from_file_location("llama_numpy", V0 / "reference" / "llama_numpy.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/p05-fixtures")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
f32 = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731

# --- RMSNorm: 4 rows of 64 ---
x, w = f32(4, 64), (1 + 0.1 * f32(64)).astype(np.float32)
fx = {"rmsnorm_x": x, "rmsnorm_w": w, "rmsnorm_out": ref.rmsnorm(x, w, 1e-5).astype(np.float32)}

# --- RoPE: 4 heads × 16 dims at positions 0, 1, 7, 100 (theta 10000) ---
positions = np.array([0, 1, 7, 100], np.int32)
cos, sin = ref.rope_tables(16, 128, 10000.0)
q = f32(4, 4, 16)
fx["rope_pos"] = positions
fx["rope_in"] = q
fx["rope_out"] = np.stack([ref.apply_rope(q[i], cos[p], sin[p]) for i, p in enumerate(positions)]).astype(np.float32)

# --- Attention: n_heads 4, n_kv_heads 2, hd 16, cache of 10 positions; output for every pos in 0..9 ---
T, H, KVH, HD = 10, 4, 2, 16
kc, vc, qs = f32(T, KVH * HD), f32(T, KVH * HD), f32(T, H, HD)
outs = []
for pos in range(T):
    K, V = kc[: pos + 1].reshape(pos + 1, KVH, HD), vc[: pos + 1].reshape(pos + 1, KVH, HD)
    o = np.stack([ref.softmax((K[:, h // (H // KVH)] @ qs[pos, h]) / np.sqrt(HD)) @ V[:, h // (H // KVH)] for h in range(H)])
    outs.append(o.reshape(-1))
fx.update({"attn_q": qs, "attn_k": kc, "attn_v": vc, "attn_out": np.stack(outs).astype(np.float32)})
save_file(fx, str(out / "ops.safetensors"))

# --- Full forward on a tiny model ---
tiny = out / "tiny"
subprocess.run([sys.executable, str(V0 / "tools" / "make_tiny_llama.py"), str(tiny)], check=True)
ids = np.array([5, 17, 3, 200, 42, 42, 7, 1, 99, 128, 64, 2], np.int32)
m = ref.LlamaNumpy(tiny)
logits = np.stack([m.forward(int(t), p) for p, t in enumerate(ids)]).astype(np.float32)
prompt = [1, 2, 3, 4]
toks, glog = ref.LlamaNumpy(tiny).generate_greedy(prompt, 48)
gaps = np.array([np.diff(np.sort(l)[-2:])[0] for l in glog], np.float32)  # top-1 minus top-2 per step
save_file({"tf_ids": ids, "tf_logits": logits, "greedy_prompt": np.array(prompt, np.int32),
           "greedy_tokens": np.array(toks, np.int32), "greedy_gaps": gaps}, str(out / "forward.safetensors"))
print("wrote", out)
