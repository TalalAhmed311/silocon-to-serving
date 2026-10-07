# 65: Gaussian Error Gated Linear Unit (L4, core; stands in for plain GELU)

Problem: [LeetGPU #65](../../leetgpu-map.md). `out = gelu(a) ⊙ b`.

**Hint ladder**

1. Exact GELU is `x·Φ(x) = ½x(1 + erf(x/√2))`. CUDA's `erff` is accurate to a couple of ulp.
2. The common **tanh approximation** `½x(1 + tanh(√(2/π)(x + 0.044715x³)))` differs from exact by up to ~1e-3 absolute. If the problem expects one and you use the other, a tight tolerance fails. Read the statement, and know the difference.
3. Same memory pattern as #54: grid-stride, and vectorize if you like.

**Solution outline:** `geglu_k` with `erff`.

**Why it matters:** gated activations are the MLP's elementwise core. GELU, SiLU and their gated forms are memory-bound unless fused into the GEMM epilogue.
