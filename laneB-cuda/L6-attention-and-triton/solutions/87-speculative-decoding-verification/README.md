# 87: Speculative Decoding Verification (L6, core)

Problem: [LeetGPU #87](../../leetgpu-map.md). Given a draft model's proposals and both models' probabilities, decide which proposals the target accepts, and produce the corrected or bonus token.

**Hint ladder**

1. The accept rule is **exact rejection sampling** (Leviathan et al., P2.6): accept draft token x with probability `min(1, p(x)/q(x))`. At the first rejection, sample from `normalize(max(0, p − q))`. If all k are accepted, sample a bonus token from the target's next distribution. The output distribution then equals the target's exactly.
2. The accept chain is **sequential** (stop at the first rejection), but k ≤ 8, so one thread handles it. The expensive part is the sample over a V-sized distribution: a **block-wide** reduction (total mass) plus a chunked prefix scan to find the inverse-CDF index.
3. Pass randoms in explicitly (`u_accept`, `u_sample`). That makes the kernel deterministic and testable against a CPU reference.
4. Batch: one block per sequence. Different sequences accept different numbers of tokens: return `n_out` per sequence, and the scheduler (P6.2) advances each sequence by its own count.

**Solution outline:** thread 0 runs the accept loop. Then `sample_block` (a block sum of weights → a chunked scan → the smallest j with CDF > u·total) on the residual or the bonus distribution.

**Why it matters:** this runs on every speculative step, and doing it on the GPU avoids copying `(k+1)·V` probabilities to the host (P6.5).
