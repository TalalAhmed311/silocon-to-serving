# 8 — Matrix Addition (L1)

Problem: [LeetGPU #8](../../leetgpu-map.md). C = A + B for N×N row-major matrices.

**Hint ladder**

1. A contiguous row-major matrix is just an `N·N`-element vector. Do you need 2-D indexing at all?
2. A 1-D grid-stride loop over `N*N` elements is the simplest correct answer. Use `size_t` for the index, since N = 46341 already overflows `int`.
3. A 2-D version (as in #31) is good practice. Make sure x maps to columns.

**Solution outline:** treat the matrix as a flat vector and run a grid-stride `C[i] = A[i] + B[i]`.

**Why this is fast:** identical to Vector Addition, 12 bytes/element, memory-bound and coalesced.
