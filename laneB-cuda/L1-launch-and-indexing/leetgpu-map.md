# LeetGPU map — L1 — Launch, indexing, bounds

Challenge titles and difficulty come from [`AlphaGPU/leetgpu-challenges@37a1253`](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646), which is CC BY-NC-ND 4.0. We **do not** reproduce problem statements, starters or tests. Open each problem on https://leetgpu.com/challenges (the per-problem site URL is UNVERIFIED; the repo link below is verified).

Priority: **core** problems are required, **practice** problems are recommended, **stretch** and **hard** problems are optional.

| # | Challenge | Difficulty | Concept tested | Priority | Hint ladder | Solution |
|---|---|---|---|---|---|---|
| 1 | [Vector Addition](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/1_vector_add) | easy | 1D grid-stride index, bounds check, error-check macro | core | TODO | TODO |
| 21 | [ReLU](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/21_relu) | easy | elementwise map; `fmaxf` | core | TODO | TODO |
| 23 | [Leaky ReLU](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/23_leaky_relu) | easy | branch vs `fmaxf`/select | core | TODO | TODO |
| 7 | [Color Inversion](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/7_color_inversion) | easy | byte-level access on `unsigned char` RGBA, `uchar4` view | core | TODO | TODO |
| 19 | [Reverse Array](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/19_reverse_array) | easy | in-place swap, half the threads, race-free indexing | core | TODO | TODO |
| 31 | [Matrix Copy](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/31_matrix_copy) | easy | 2D indexing, row-major `row*N+col` | core | TODO | TODO |
| 8 | [Matrix Addition](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/8_matrix_addition) | easy | 2D elementwise | core | TODO | TODO |
| 66 | [RGB to Grayscale](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/66_rgb_to_grayscale) | easy | AoS pixel layout, 3-channel stride | core | TODO | TODO |
| 62 | [Value Clipping](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/62_value_clipping) | easy | clamp, branchless | practice | TODO | TODO |
| 63 | [Interleave Arrays](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/63_interleave) | easy | output index ≠ input index (scatter) | practice | TODO | TODO |
| 68 | [Sigmoid Activation](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/68_sigmoid) | easy | `expf` and fast-math accuracy | practice | TODO | TODO |
| 52 | [Sigmoid Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/52_silu) | easy | fused activation `x*sigmoid(x)` | practice | TODO | TODO |
| 2 | [Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/2_matrix_multiplication) | easy | naive matmul: one thread per output, 2D grid for any M×N×K | core (exit check) | TODO | TODO |
| 41 | [Simple Inference](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/41_simple_inference) | easy | matmul + bias as a linear layer | stretch | TODO | TODO |
| 24 | [Rainbow Table](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646/challenges/easy/24_rainbow_table) | easy | per-thread loop of integer hashing | stretch | TODO | TODO |
