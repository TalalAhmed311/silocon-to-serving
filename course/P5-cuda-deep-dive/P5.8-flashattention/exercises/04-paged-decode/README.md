# Exercise 4 (hard): decode attention over a paged KV cache (T2, feeds P6.3)

During decode, each sequence contributes **one** query per step, but its KV history is spread across fixed-size **blocks** in a shared pool. That's PagedAttention (P2.3, P6.3). The kernel gets a `block_table[b][i]` mapping logical block i of sequence b to a physical block.

Implement `paged_decode` in `kernel.cuh`:

- one block per (sequence, query head). GQA: query head `h` reads KV head `h / (H / Hkv)`
- warps walk the sequence's tokens (strided), each keeping its own online-softmax state `(m, l, o)`. Lanes split the 64 dims (2 each), and a warp shuffle reduction gives each token's score
- merge the warps' states with the online-softmax rule (P5.5) at the end

The test builds the paged cache with deliberately scrambled physical blocks, gathers each sequence back into dense form, and compares with the dense reference for the last query position. It covers lengths 1, 15/16/17 (block edges) and GQA. P6.3 calls this kernel from the #0 v1 engine.
