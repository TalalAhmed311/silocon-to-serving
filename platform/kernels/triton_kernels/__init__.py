"""triton_kernels — the D4 kernels rewritten in Triton (P5.9): softmax, RMSNorm, matmul, FlashAttention-2 forward.

Every kernel runs on a GPU, and also in Triton's interpreter (TRITON_INTERPRET=1) on CPU tensors for T0 tests.
Pinned: triton v3.8.0 (SOURCES.md). API names (tl.dot input_precision, autotune) per that version.
"""
