# P1.2 quiz

<details><summary><b>1.</b> Why is prefill compute-bound and decode memory-bound on the same GPU?</summary>

Prefill reuses each weight across T tokens (a GEMM, intensity ≈ T). Batch-1 decode uses each weight once per read (a GEMV, intensity ≈ 1).
</details>

<details><summary><b>2.</b> KV bytes per token for 40 layers, 8 KV heads, head_dim 128, bf16?</summary>

2 × 40 × 8 × 128 × 2 = **163,840 B = 160 KiB**.
</details>

<details><summary><b>3.</b> What usually limits the number of concurrent sequences: weights or KV cache?</summary>

The **KV cache**. Weights are a fixed cost, while KV grows with every sequence × context length.
</details>

<details><summary><b>4.</b> Why doesn't batching make long-context decode compute-bound?</summary>

Each sequence reads its own KV cache, so those bytes grow with B just like the FLOPs. As B → ∞ the intensity tends to a constant set by the attention work per KV byte, about 1 FLOP/byte at fp16.
</details>

<details><summary><b>5.</b> What does FP8 KV cache buy, and what does it cost?</summary>

2× capacity (more sequences or longer context) and 2× fewer KV bytes read per step. The cost is quantization error in attention, which must be measured (P2.5).
</details>

<details><summary><b>6.</b> Without a KV cache, how many forward-pass token-steps does generating n tokens cost?</summary>

1 + 2 + … + n = n(n+1)/2, which is O(n²), instead of n.
</details>

<details><summary><b>7.</b> Which latency metric does prefill determine, and which does decode determine?</summary>

Prefill determines **TTFT**. Decode determines **ITL/TPOT**.
</details>
