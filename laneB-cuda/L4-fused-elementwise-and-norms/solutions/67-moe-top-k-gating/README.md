# 67: MoE Top-K Gating (L4, stretch: hints and outline only)

Problem: [LeetGPU #67](../../leetgpu-map.md). For each token, softmax over E expert logits, pick the top k experts, and renormalize their weights.

**Hint ladder**

1. E is small (8–256), so give **one warp per token**: each lane holds E/32 logits in registers.
2. Softmax with warp shuffles (#5 in miniature).
3. Top-k with small k (1–8): k rounds of warp argmax (`__shfl_xor_sync` on (value, index) pairs, ties to the lower index), masking each winner out. That's cheaper than sorting.
4. Renormalize the k selected weights (or not: models differ, check the statement). Also emit **per-expert token counts** (atomics in shared memory), because the next step (permuting tokens by expert for grouped GEMMs) needs them.

**Solution outline:** warp per token: logits → shuffle softmax → k × shuffle argmax → write (expert ids, weights) and bump the counts. Connects to P4.2 (expert parallelism, all-to-all) and Silicon to Scale ch. 13.
