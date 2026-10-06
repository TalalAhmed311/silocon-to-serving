# LeetGPU map — L3 — Tree reduction, warp shuffles, scan, cooperative groups

Challenge titles and difficulty come from [`AlphaGPU/leetgpu-challenges@37a1253`](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646), which is CC BY-NC-ND 4.0. We **do not** reproduce problem statements, starters or tests. Open each problem on https://leetgpu.com/challenges (the per-problem site URL is UNVERIFIED; the repo link below is verified).

Priority: **core** problems are required, **practice** problems are recommended, **stretch** and **hard** problems are optional.

| # | Challenge | Difficulty | Concept tested | Priority | Hint ladder | Solution |
|---|---|---|---|---|---|---|
| 4 | [Reduction](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/4_reduction) | medium | tree reduction → `__shfl_down_sync` → grid-stride + one atomic | core (exit check vs `cub::DeviceReduce`) | TODO | TODO |
| 17 | [Dot Product](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/17_dot_product) | medium | map + reduce fused | core | TODO | TODO |
| 27 | [Mean Squared Error](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/27_mean_squared_error) | medium | fused squared diff + reduce | practice | TODO | TODO |
| 16 | [Prefix Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/16_prefix_sum) | medium | Blelloch / decoupled look-back scan | core | TODO | TODO |
| 70 | [Segmented Exclusive Prefix Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/70_segmented_prefix_sum) | medium | segmented scan with flags | practice | TODO | TODO |
| 72 | [Stream Compaction](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/72_stream_compaction) | medium | scan-based compaction | core | TODO | TODO |
| 47 | [Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/47_subarray_sum) | medium | prefix-sum trick for range sums | practice | TODO | TODO |
| 48 | [2D Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/48_2d_subarray_sum) | medium | 2D variant | practice | TODO | TODO |
| 49 | [3D Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/49_3d_subarray_sum) | medium | 3D variant | stretch | TODO | TODO |
| 51 | [Max Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/51_max_subarray_sum) | medium | scan of (sum, max) pairs | stretch | TODO | TODO |
| 29 | [Top K Selection](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/29_top_k_selection) | medium | per-block top-k + merge | core | TODO | TODO |
| 15 | [Sorting](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/15_sorting) | hard | bitonic sort in shared memory | core | TODO | TODO |
| 36 | [Radix Sort](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/hard/36_radix_sort) | hard | radix sort with scan of digit histograms | stretch | TODO | TODO |
| 71 | [Parallel Merge](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/71_parallel_merge) | medium | merge path partitioning | stretch | TODO | TODO |
| 18 | [Sparse Matrix-Vector Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/18_sparse_matrix_vector_multiplication) | medium | CSR SpMV, warp per row | core | TODO | TODO |
| 75 | [Sparse Matrix-Dense Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/75_sparse_matrix_dense_matrix_multiplication) | medium | CSR × dense | stretch | TODO | TODO |
| 82 | [Linear Recurrence](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/82_linear_recurrence) | medium | scan with an associative linear operator | stretch | TODO | TODO |
| 110 | [Parallel Reverse Scan (GAE)](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/110_gae_reverse_scan) | medium | reverse scan | stretch | TODO | TODO |
| 38 | [Nearest Neighbor](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/38_nearest_neighbor) | medium | brute-force nearest neighbor with block reduction | practice | TODO | TODO |
| 35 | [Monte Carlo Integration](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/35_monte_carlo_integration) | medium | parallel RNG + reduction | practice | TODO | TODO |
