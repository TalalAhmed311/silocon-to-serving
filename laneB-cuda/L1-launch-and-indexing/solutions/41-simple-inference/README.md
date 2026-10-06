# 41 — Simple Inference (L1, stretch)

Problem: [LeetGPU #41](../../leetgpu-map.md). A single linear layer applied to a batch: `y = x · Wᵀ + b`. Read the statement for the exact shapes and argument order. Our harness uses `x: [batch, in]`, `W: [out, in]` (the PyTorch `nn.Linear` layout), `b: [out]`, `y: [batch, out]`.

**Hint ladder**

1. It is the naive matmul from #2 with one extra add. Which output element does each thread own?
2. Thread `(r, o)` computes `Σᵢ x[r][i] · W[o][i] + b[o]`. Note `W` is indexed `[o][i]`: you dot two **rows**.
3. Map `threadIdx.x → o` so the warp's `W` reads… are they coalesced? (No: lanes read different rows of W. Each lane's own loop over `i` *is* sequential, so the L1 cache helps. L2's tiling fixes it properly.)

**Solution outline:** the #2 kernel with `W` read as `W[o*in + i]`, plus the bias.

**Why this matters:** this is P0.5's matvec on the GPU, batched. At `batch = 1` it is memory-bound exactly as the roofline predicted. At large batch it becomes a GEMM.
