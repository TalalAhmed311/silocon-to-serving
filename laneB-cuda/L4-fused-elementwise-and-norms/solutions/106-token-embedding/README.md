# 106: Token Embedding Layer (L4, practice)

Problem: [LeetGPU #106](../../leetgpu-map.md). Gather rows of the embedding table by token id.

**Hint ladder**

1. Pure memory movement. The gather is **by row**, so each row copy is contiguous and coalesced.
2. One block per token, threads striding over D, `float4` when `D % 4 == 0`.
3. Decide what out-of-range ids do (zero, clamp, or trap) and match the statement.
4. In a real engine the embedding lookup reads only T rows of a `V × D` table. That's why P1.1 excludes the embedding from the "weights read per token" count while including the LM head.

**Solution outline:** `embed_k`: an id check, then a vectorized row copy.

**Why this is fast:** it moves exactly `T·D·4` bytes twice (read + write). Nothing to optimize beyond full transactions.
