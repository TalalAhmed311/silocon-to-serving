# CuTe HGEMM: a guided path (exercise 4, hard)

CuTe (part of CUTLASS) describes tensors as **layouts**: shape + stride functions composed hierarchically. Tiling, partitioning across threads, and the MMA/copy atoms are all layout algebra. It's how CUTLASS 3/4 kernels are written, and it's worth learning once you've hand-written `mma.sync` and seen how error-prone fragment indexing is.

The course doesn't vendor CUTLASS. Clone it at the pinned tag and build its tutorial first:

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass.git     # tag per SOURCES.md
# read, in order (paths per the v4.8.0 tree; verified at build time):
#   media/docs/cpp/cute/01_layout.md, 02_layout_algebra.md, 03_tensor.md, 04_algorithms.md, 0t_mma_atom.md,
#   0x_gemm_tutorial.md
#   examples/cute/tutorial/  (sgemm_1.cu → sgemm_2.cu → sgemm_sm80.cu)
```

Then write `04_cute_hgemm.cu` yourself:

1. Start from the tutorial's sm80 GEMM: `TiledMMA` with `SM80_16x8x16_F32F16F16F32_TN`, `TiledCopy` with `SM80_CP_ASYNC_CACHEGLOBAL`, and `ldmatrix` via `SM75_U32x4_LDSM_N`.
2. Match **our** problem: row-major A (M×K) and B (K×N) with fp32 C. The tutorial's TN layout means B is K-major. Either transpose B once, or change B's smem layout and copy atom (that's the real exercise).
3. Build it in this repo's CMake by adding the CUTLASS `include/` and `tools/util/include/` directories to `INC` (a local edit; don't commit the clone).
4. Test it against `d4::hgemm_cublas` with the same tolerance as `test_hgemm.cu`, and bench it next to the hand-written variants.

Exit check (L5): **≥ 50% of cuBLAS**. `TODO(run-on: g6.xlarge)`. Write down which CuTe abstraction replaced which 10 lines of your hand-written `mma.sync` kernel.
