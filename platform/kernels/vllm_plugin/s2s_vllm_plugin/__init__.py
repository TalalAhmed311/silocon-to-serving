"""s2s_vllm_plugin — replace vLLM's fused_add_rms_norm with ours behind a flag (#12).

vLLM's RMSNorm layer (vllm/model_executor/layers/layernorm.py at v0.31.0) calls
`ops.fused_add_rms_norm(x, residual, weight, eps)` where `ops` is `vllm._custom_ops`; that function forwards to
`torch.ops._C.fused_add_rms_norm` (csrc/layernorm_kernels.cu). We patch the Python attribute, so the layer's call site
(an attribute lookup at call time) reaches our registered custom op instead. Names/paths verified against the pinned
vLLM SHA for the course; re-check after upgrading (the layer may also take a different, fused-quant path).

Enable with S2S_RMSNORM=1 (default off, so installing the plugin changes nothing until you opt in).
"""
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def register():
    if os.environ.get("S2S_RMSNORM") != "1":
        return
    here = Path(__file__).resolve()
    sys.path[:0] = [str(here.parents[2]), str(here.parents[3])]           # platform/kernels (torch_ext), platform (kernels.*)
    import torch

    import torch_ext.ops  # noqa: F401  registers torch.ops.s2s.fused_add_rms_norm
    from vllm import _custom_ops as vops

    def ours(input, residual, weight, epsilon):  # noqa: A002 — vLLM's parameter names
        torch.ops.s2s.fused_add_rms_norm(input, residual, weight, epsilon)

    vops.fused_add_rms_norm = ours
    log.warning("s2s plugin: fused_add_rms_norm → torch.ops.s2s.fused_add_rms_norm (impl=%s)",
                os.environ.get("S2S_RMSNORM_IMPL", "cuda"))
