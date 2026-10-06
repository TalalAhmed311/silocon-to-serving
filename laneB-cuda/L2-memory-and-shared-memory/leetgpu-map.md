# LeetGPU map — L2 — Coalescing, shared memory, atomics, occupancy

Challenge titles and difficulty come from [`AlphaGPU/leetgpu-challenges@37a1253`](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646), which is CC BY-NC-ND 4.0. We **do not** reproduce problem statements, starters or tests. Open each problem on https://leetgpu.com/challenges (the per-problem site URL is UNVERIFIED; the repo link below is verified).

Priority: **core** problems are required, **practice** problems are recommended, **stretch** and **hard** problems are optional.

| # | Challenge | Difficulty | Concept tested | Priority | Hint ladder | Solution |
|---|---|---|---|---|---|---|
| 3 | [Matrix Transpose](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/3_matrix_transpose) | easy | coalesced read + write through a padded shared tile (`[32][33]`) | core (exit check) | TODO | TODO |
| 9 | [1D Convolution](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/9_1d_convolution) | easy | halo loading into shared memory, `__constant__` kernel weights | core | TODO | TODO |
| 10 | [2D Convolution](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/10_2d_convolution) | medium | 2D tiles with halo | core | TODO | TODO |
| 11 | [3D Convolution](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/11_3d_convolution) | medium | 3D tiling; register reuse along z | stretch | TODO | TODO |
| 28 | [Gaussian Blur](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/28_gaussian_blur) | medium | separable vs direct 2D filter | practice | TODO | TODO |
| 42 | [2D Max Pooling](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/42_2d_max_pooling) | medium | 2D windowed reduction | practice | TODO | TODO |
| 69 | [2D Jacobi Stencil](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/69_jacobi_stencil_2d) | medium | stencil with shared-memory reuse | practice | TODO | TODO |
| 13 | [Histogramming](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/13_histogramming) | medium | privatized shared-memory histograms + global atomics | core | TODO | TODO |
| 43 | [Count Array Element](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/43_count_array_element) | medium | atomics vs block-local counting | core | TODO | TODO |
| 44 | [Count 2D Array Element](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/44_count_2d_array_element) | medium | 2D variant | practice | TODO | TODO |
| 45 | [Count 3D Array Element](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/45_count_3d_array_element) | medium | 3D variant | practice | TODO | TODO |
| 2 | [Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/2_matrix_multiplication) | easy | revisit: shared-memory tiled matmul | core | TODO | TODO |
| 37 | [Matrix Power](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/medium/37_matrix_power) | medium | repeated tiled matmul, launch overhead | stretch | TODO | TODO |
