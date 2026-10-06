# 19 — Reverse Array (L1)

Problem: [LeetGPU #19](../../leetgpu-map.md). Reverse an array **in place**.

**Hint ladder**

1. If every thread swaps `a[i]` with `a[N−1−i]`, what happens when thread `i` and thread `N−1−i` both run?
2. Each pair must be swapped exactly once, so launch only `N/2` threads.
3. If N is odd, the middle element stays put: `N/2` threads (integer division) is already right.

**Solution outline:** threads `i < N/2` swap `a[i]` and `a[N−1−i]`, with no synchronization. Each pair is owned by exactly one thread, so there is no race.

**Why this is fast:** both the front reads and the mirrored back reads are contiguous within a warp. The back addresses decrease, but the 32 lanes still cover one contiguous 128-byte span, so they coalesce. That is 2 reads + 2 writes per pair, memory-bound.
