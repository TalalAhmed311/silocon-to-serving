"""torch_ext — #12: our fused residual-add + RMSNorm as a PyTorch custom op (s2s::fused_add_rms_norm), with a CUDA
implementation (JIT-built extension over d4/norms.cuh logic) and a Triton implementation, plus the vLLM plugin that
swaps it in. Import `torch_ext.ops` to register the op."""
