# P1.4 quiz

<details><summary><b>1.</b> A datasheet says "FP8 Tensor: 1,979 TFLOPS*" (*with sparsity). What do you use for a dense model?</summary>

≈ **990 TFLOPS** (half).
</details>

<details><summary><b>2.</b> An L4 has ~300 GB/s and ~121 dense fp16 TFLOP/s. Ridge point? Is batch-64 decode compute-bound?</summary>

Ridge ≈ 403 FLOP/byte. Batch-64 decode is ≈ 64 FLOP/byte, so it is **memory-bound**.
</details>

<details><summary><b>3.</b> Ceiling for batch-1 decode of a 7B fp16 model on a GPU with 2 TB/s?</summary>

14 GB per token, so 2e12 / 14e9 ≈ **143 tokens/s**.
</details>

<details><summary><b>4.</b> How does a GPU SM hide a 400-cycle memory latency without a big cache?</summary>

It keeps many warps resident and switches to a ready warp each cycle. While some warps wait on memory, others compute.
</details>

<details><summary><b>5.</b> Why measure bandwidth with a 1 GiB buffer and not 16 MiB on an L4?</summary>

The L4's L2 (~48 MB) would serve a 16 MiB buffer from cache. You'd measure L2 bandwidth, not GDDR.
</details>

<details><summary><b>6.</b> T4 vs L4 for an FP8 quantization bakeoff?</summary>

L4. The T4 (Turing) has no FP8 tensor cores (and no bf16).
</details>

<details><summary><b>7.</b> Which ops should you fuse, and why?</summary>

The memory-bound elementwise ops (norms, activations, residual adds). Each one costs a full read and write of the activations for almost no FLOPs, so fusing removes HBM round trips (P5.5).
</details>
