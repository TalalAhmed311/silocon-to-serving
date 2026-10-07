# 25: Categorical Cross Entropy Loss (L4, core)

Problem: [LeetGPU #25](../../leetgpu-map.md). Mean over rows of `−log softmax(logits)[label]`.

**Hint ladder**

1. Never compute `log(softmax(x)[y])`. For confident predictions the softmax underflows to 0. Use `logsumexp(x) − x[y]`.
2. `logsumexp = m + log(Σ e^{x−m})`: the online `(m, d)` pass from #5 gives it in one read.
3. The thread that sees column `label` saves the logit in shared memory. The block reduction's `__syncthreads` makes it visible.
4. Reduce across rows with one `atomicAdd(loss/N)` per row, or a second pass if you need determinism.

**Solution outline:** one block per row, online max/sum + label gather → block merge → `atomicAdd((M + log D − x_y)/N)`.

**Why this is fast:** one read of the logits, which is the largest tensor in an LM's forward pass at large vocabularies. P5.5 exercise 4 is the same kernel at a 128k vocabulary.
