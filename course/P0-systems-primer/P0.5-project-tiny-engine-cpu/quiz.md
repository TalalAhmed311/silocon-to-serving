# P0.5 quiz

<details><summary><b>1.</b> Why does a KV cache turn per-token attention work from O(t²) into O(t)?</summary>

Without it, step t recomputes K and V for all t previous tokens, which costs O(t) per step and O(t²) in total. With it, each step computes K and V for one token and only *reads* the stored ones. The attention scores are still O(t) per step.
</details>

<details><summary><b>2.</b> A model has 32 query heads and 8 KV heads. Which KV head does query head 13 use, and how much smaller is the cache than with full MHA?</summary>

group = 32/8 = 4, so head 13 uses KV head 13/4 = **3**. The cache is **4× smaller**.
</details>

<details><summary><b>3.</b> Your outputs are fine at position 0 and garbage from position 1 on. Most likely bug?</summary>

RoPE. At pos 0 the rotation is the identity, so any convention "works". Check rotate_half vs interleaved, and that you rotate both q and k (not v).
</details>

<details><summary><b>4.</b> Why is decode at batch 1 memory-bound, and what does that predict for tokens/s?</summary>

Each weight is read once per token and used for one multiply-add, so the intensity is ~0.5 FLOP/byte in fp32, far below the ridge. tokens/s ≲ bandwidth / weight bytes.
</details>

<details><summary><b>5.</b> Top-p = 0.9 over probabilities {0.6, 0.25, 0.1, 0.05}. Which tokens can be sampled, with what probabilities?</summary>

The cumulative masses are 0.6, 0.85, 0.95 ≥ 0.9, so the first three are kept. Renormalized over 0.95: **0.632, 0.263, 0.105**.
</details>

<details><summary><b>6.</b> What does temperature T → 0 do, and why does the engine special-case T ≤ 0?</summary>

It sharpens the distribution toward one-hot on the argmax. T = 0 would divide by zero, so it is defined as greedy.
</details>

<details><summary><b>7.</b> Why do the engine tests compare <i>teacher-forced logits</i> rather than only generated tokens?</summary>

Tokens are a lossy summary: a tiny numeric difference can flip an argmax at a near-tie, after which the sequences legitimately diverge. Teacher-forced logits compare every step's full output on the same inputs, with an explicit tolerance.
</details>

<details><summary><b>8.</b> Int8 weights cut weight bytes 4×. Why might the tiny test model show no speedup?</summary>

Its weights already fit in the CPU caches, so it is limited by thread-pool overhead and cache bandwidth, not by DRAM bytes. The roofline's B depends on where the data lives.
</details>
