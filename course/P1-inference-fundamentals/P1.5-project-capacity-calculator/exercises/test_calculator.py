import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors import safe_open

from s2s.exercise import load_impl

ROOT = Path(__file__).resolve().parents[4]


def _load():
    # calculator.py lives directly in exercises/ (not a subfolder), so load it by path.
    import importlib.util
    import os
    here = Path(__file__).resolve().parent
    path = here / "solutions" / "calculator.py" if os.environ.get("S2S_SOLUTIONS") == "1" else here / "calculator.py"
    spec = importlib.util.spec_from_file_location("p15_calculator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()
PRESETS = ROOT / "platform/capacity/presets"
GPU24 = {"name": "fixture-24", "memory_gb": 24, "hbm_gbs": 300}


@pytest.mark.parametrize("tie", [False, True])
def test_weights_bytes_exact_vs_checkpoint(tmp_path, tie):
    args = [sys.executable, str(ROOT / "platform/engine/v0/tools/make_tiny_llama.py"), str(tmp_path)] + (["--tie"] if tie else [])
    subprocess.run(args, check=True, capture_output=True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    with safe_open(str(tmp_path / "model.safetensors"), "np") as f:
        actual = sum(math.prod(f.get_slice(k).get_shape()) * 4 for k in f.keys())
    assert m.weights_bytes(cfg, "fp32") == actual


def test_kv_and_capacity():
    cfg = json.loads((PRESETS / "llama3-8b.json").read_text())
    assert m.kv_bytes_per_token(cfg, "bf16") == 131072
    assert m.kv_bytes_per_token(cfg, "fp8") == 65536
    # 24 GiB × 0.9 − 8.03e9×2 B − 1 GiB ≈ 5.6 GiB → / (4096 × 128 KiB) ≈ 11 sequences
    n = m.max_sequences(cfg, GPU24, "bf16", "bf16", 4096)
    assert 8 <= n <= 13
    assert m.max_sequences(cfg, GPU24, "fp8", "fp8", 4096) > 2 * n   # fp8 halves weights AND KV


def test_parse_vllm_log():
    sample = "INFO 10-06 12:00:00 [kv_cache_utils.py] GPU KV cache size: 45,056 tokens\nINFO ... Maximum concurrency ..."
    assert m.parse_vllm_kv_tokens(sample) == 45056
    assert m.parse_vllm_kv_tokens("nothing here") is None


def test_moe_and_tp():
    cfg = json.loads((PRESETS / "mixtral-8x7b.json").read_text())
    total, active = m.total_params(cfg), m.active_params(cfg)
    # Hand derivation (P1.5 lesson): per layer attn 41.9M + 8 experts × 176.2M + router + norms; ×32; + embed + head
    assert total == pytest.approx(46.70e9, rel=0.01)
    assert active == pytest.approx(12.75e9, rel=0.01)       # 2 of 8 experts + attention + head (embedding excluded)
    assert m.weights_bytes(cfg, "bf16", tp=2) == pytest.approx(m.weights_bytes(cfg, "bf16") / 2)
    assert m.kv_bytes_per_token(cfg, "bf16", tp=2) == pytest.approx(m.kv_bytes_per_token(cfg, "bf16") / 2)
