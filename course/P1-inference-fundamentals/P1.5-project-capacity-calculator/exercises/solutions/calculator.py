"""P1.5 reference: thin wrappers over platform/capacity/core.py so the tests exercise the real tool."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "platform"))
from capacity.core import DTYPE_BYTES, Model  # noqa: E402,F401


def _m(cfg):
    return Model.from_config(cfg)


def total_params(cfg):
    return _m(cfg).total_params()


def active_params(cfg):
    return _m(cfg).active_params()


def weights_bytes(cfg, dtype, tp=1):
    return total_params(cfg) * DTYPE_BYTES[dtype] / tp


def kv_bytes_per_token(cfg, kv_dtype, tp=1):
    return _m(cfg).kv_bytes_per_token(DTYPE_BYTES[kv_dtype]) / tp


def max_sequences(cfg, gpu, weight_dtype, kv_dtype, context, mem_util=0.9, activation_gb=1.0, tp=1):
    budget = gpu["memory_gb"] * 2**30 * mem_util - weights_bytes(cfg, weight_dtype, tp) - activation_gb * 2**30
    return max(0, int(budget // (kv_bytes_per_token(cfg, kv_dtype, tp) * context)))


def parse_vllm_kv_tokens(log_text):
    # Expected (UNVERIFIED) vLLM v1 start-up line:  "GPU KV cache size: 123,456 tokens"
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log_text)
    return int(m.group(1).replace(",", "")) if m else None
