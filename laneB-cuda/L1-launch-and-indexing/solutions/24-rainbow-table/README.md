# 24 — Rainbow Table (L1, stretch)

Problem: [LeetGPU #24](../../leetgpu-map.md). Apply a hash function R times to every input integer, independently per element.

> **Use the exact hash function and round count from the problem statement.** It defines them, and we don't reproduce it. Our harness uses 32-bit FNV-1a over the 4 bytes of the value as a stand-in (`hash_round` in `kernel.cu`). Swap in the statement's function before submitting.

**Hint ladder**

1. Every element is independent, so it is one thread per element again. The new part is the per-thread *loop*.
2. Keep the running value in a register. Load once, run `R` rounds, store once.
3. This problem is **compute-bound**, unlike every other L1 problem. With R rounds, the work per byte grows linearly with R.

**Solution outline:** `uint32_t v = in[i]; for (r < R) v = hash_round(v); out[i] = v;`

**Why this is interesting:** it is the first problem where memory bandwidth doesn't matter. Time scales with R × N, and occupancy and integer throughput (IMAD, XOR, shifts) decide speed. Try R = 1 vs R = 1000 in `--bench` and watch the bound switch from memory to compute.
