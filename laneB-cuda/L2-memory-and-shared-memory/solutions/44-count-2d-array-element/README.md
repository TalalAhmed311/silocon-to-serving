# 44 — Count 2D Array Element (L2, practice)

Problem: [LeetGPU #44](../../leetgpu-map.md). The same task as #43 on an `R × C` matrix.

**Hint ladder**

1. A contiguous matrix is a flat array of `R·C` elements. Reuse #43's kernel.
2. If the statement passes a pitch or stride, the matrix isn't contiguous: index `row * stride + col` and use a 2-D grid-stride loop.
3. Use `size_t` for `R·C`.

**Solution outline:** delegate to the #43 kernel with `n = R·C`. The point is to *recognize* that.
