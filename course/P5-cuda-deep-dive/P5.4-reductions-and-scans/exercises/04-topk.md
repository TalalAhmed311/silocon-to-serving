# Exercise 4 (hard): top-k via per-block selection + merge (T2)

You've already met this as Lane B L3 #29. Its solution, [`laneB-cuda/L3-reductions-and-scans/solutions/29-top-k-selection`](../../../../laneB-cuda/L3-reductions-and-scans/solutions/29-top-k-selection/README.md), uses tournament rounds of in-shared-memory bitonic sorts and leaves `k > 1024` as an exercise.

Here, build the **reduction-style** alternative and compare:

1. Each warp keeps a sorted top-k in registers (k ≤ 32: one value per lane), merging candidates with shuffles. Each block merges its warps' lists in shared memory and writes one list. A final pass merges the block lists.
2. A **radix select** for large k: histogram the top 8 bits of the float keys (flip the bits so the order is monotonic), scan the histogram to find the bucket holding the k-th largest, recurse on that bucket's next 8 bits, then compact everything ≥ the threshold (L3 #72).
3. Bench `N = 2²⁴`, `k ∈ {1, 32, 1024, 50000}` for both methods and `torch.topk` if you have PyTorch on the box: `| k | tournament ms | radix-select ms | torch.topk ms |`.

Sampling (P6.5) needs top-k/top-p over a ~128k vocabulary per sequence, every decode step. Which method fits that shape?
