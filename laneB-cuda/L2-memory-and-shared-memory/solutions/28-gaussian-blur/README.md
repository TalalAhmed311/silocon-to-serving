# 28 — Gaussian Blur (L2, practice)

Problem: [LeetGPU #28](../../leetgpu-map.md). Convolve an image with a Gaussian kernel. **Check the statement** for:
- whether the kernel is given or must be computed from σ
- the output size (same or valid)
- the border handling (zero, clamp, or reflect)

Our harness uses a given `K×K` kernel, **same-size** output and **zero padding** at the borders.

**Hint ladder**

1. It is #10 (2-D convolution) with a centered kernel: the tile origin shifts by −K/2, and out-of-image loads read 0.
2. A Gaussian is **separable**: `G(x, y) = g(x)·g(y)`. Two 1-D passes (rows, then columns) cost `2K` multiplies per pixel instead of `K²`.
3. Keep the intermediate result in fp32, and check that the separable and direct versions agree within tolerance.

**Solution outline:** a direct 2-D tiled convolution with a centered halo (this file). The separable version is an exercise: write it and compare in `--bench`.

**Why it matters:** separability is the first "change the algorithm, not the kernel" optimization: K²/2K is 4.5× fewer FLOPs for K = 9.
