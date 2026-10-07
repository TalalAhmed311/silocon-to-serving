# P5.7 examples (T2)

| File | Min SM | What |
|---|---|---|
| [`hgemm_bench.cu`](hgemm_bench.cu) | 70 (WMMA), 80 (mma.sync, cp.async) | four tensor-core HGEMMs from `d4/hgemm.cuh` vs cuBLAS `GemmEx` |
| [`int8_gemm.cu`](int8_gemm.cu) | 61 | int8 GEMM with `__dp4a` vs fp32 SGEMM |
| [`cute_hgemm.md`](cute_hgemm.md) | 80 | the guided CuTe path (exercise 4) |

Kernel sources: [`platform/kernels/include/d4/hgemm.cuh`](../../../../platform/kernels/include/d4/hgemm.cuh). Read the fragment-layout comments next to `warp_mma_tile` and `store_acc` with the PTX ISA's `mma.m16n8k16` figures open.
