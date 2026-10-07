# Exercise 1: naive attention, against a reference (T2)

Implement `attention` in `kernel.cuh` the straightforward way: a kernel for `S = QKᵀ/√64` with the causal mask, a row softmax (reuse your P5.5 kernel or `d4::softmax`), and a kernel for `O = P·V`. The S workspace is passed in. At N = 4096 and B·H = 16 it's **1 GiB** of fp32. That's the cost FlashAttention removes.

The test compares with a double-precision CPU reference from the same fp16-rounded inputs. **Tolerance:** the output is rounded once to fp16 (u = 2⁻¹¹ ≈ 4.9e-4 relative, outputs ≤ 1 in magnitude), the math is fp32 with `__expf`, so `atol = 2e-3` covers it with margin.

Count the HBM traffic for N = 4096, B·H = 16: Q, K, V reads, plus S written, softmax read+write, P read. Compare with your measured time × bandwidth.

Optional cross-check: `examples/vs_flash_attn.py` runs PyTorch SDPA on the same shapes.
