# P5 — CUDA C++ deep dive

**Weeks 25–34 · Lane A ≈ 110 h · Tier T2 (one NVIDIA GPU; `g4dn.xlarge` T4 for sm_75 basics, `g6.xlarge` L4 for sm_89 FP8/bf16; Nsight needs counter access, see `aws.md`) · Lane B: L4 (25–27) → L5 (28–31) → L6 (32–) daily**

**Minimum toolchain:** CUDA 12.4+, CMake 3.24+. Every example states its minimum compute capability (sm_75 baseline; tensor-core rungs mark sm_80+ for `mma.sync` bf16 and `cp.async`, sm_89+ for FP8).

**Thread through the phase:** you learn the GPU by climbing ladders. Every kernel goes from naive to fast in measured rungs, each one profiled with Nsight Compute and placed on the roofline. All of them land in the **D4 kernels repo**. The phase ends by swapping one hot path in the serving stack for your own kernel (#12).

**Spine.** PMPP 4th ed. (chapters assigned per module) + its authors' lectures · CUDA C++ Programming / Best Practices Guides · siboehm SGEMM_CUDA · GPU MODE lectures.

**Shared harness.** `laneB-cuda/harness/` is a CMake + CTest harness: `check(kernel, reference, rtol, atol)` against a CPU or cuBLAS/CUB reference, plus `bench(kernel)` with warm-up, N reps, and median/p90 via CUDA events. It prints `size | time | GB/s or TFLOP/s | % of peak` and writes JSON. P5 examples and Lane B solutions both use it. Peak values come from `P1.4/gpu_specs.yaml` (cited) and from a measured copy/FMA microbenchmark.

| Module | Time | Project | Animations |
|---|---|---|---|
| P5.1 Execution model, occupancy, divergence | 0.5 h + 9 h | — | **grid → blocks → warps → SM scheduling; warp divergence**; handbook GPUExecutionVisualizer, WarpDivergenceVisualizer, WarpSchedulerVisualizer (linked) |
| P5.2 Memory: coalescing, shared memory, bank conflicts | 0.5 h + 10 h | — | **coalescing: contiguous vs strided, sectors fetched**; **shared-memory tiling of matmul; bank conflicts with/without padding**; handbook MemoryCoalescingVisualizer (linked; ours adds a sector count) |
| P5.3 Profiling with Nsight Compute and Systems | 0.4 h + 8 h | — | — |
| P5.4 Reductions and scans | 0.5 h + 10 h | D4 (reduction, scan) | **tree reduction vs warp-shuffle reduction** |
| P5.5 Softmax and norms (fused, vectorized) | 0.5 h + 10 h | D4 (softmax, RMSNorm) | **online softmax (running max and sum)**; handbook KernelFusionVisualizer (linked) |
| P5.6 The GEMM ladder (rungs 1–7) | 0.7 h + 16 h | D4 (GEMM ladder) | reuses the P5.2 tiling animation |
| P5.7 Tensor cores: WMMA → `mma.sync` → CuTe (rung 8) | 0.6 h + 14 h | D4 (HGEMM, int8 matmul) | — |
| P5.8 FlashAttention v1 → v2 forward | 0.6 h + 14 h | D4 (FA-2 forward) | **FlashAttention tiling over Q/K/V blocks** |
| P5.9 Triton | 0.4 h + 9 h | D4 (Triton rewrites) | — |
| P5.10 Wiring a custom op into PyTorch/vLLM | 0.4 h + 10 h | **#12 custom kernel path** | — |

---

## P5.1 — Execution model, occupancy, divergence

**Objectives.** (1) The thread → warp → block → grid mapping, and how blocks are scheduled onto SMs. (2) Occupancy: registers, shared memory and block size, using the occupancy API and `--resource-usage`. (3) Warp divergence and predication. (4) Error checking, streams, async launches and `cudaDeviceSynchronize` pitfalls. (5) Pinned vs pageable `cudaMemcpy` bandwidth, closing the loop with P0.2.

**Examples.** `01_hello_indices.cu` · `02_occupancy.cu` · `03_divergence.cu` (naive vs branch-free) · `04_pinned_vs_pageable.cu` · `05_streams_overlap.cu`.

**Exercises.** (1) Grid-stride vector add for any N (harness test) · (2) pick a block size by occupancy and justify it with the numbers · (3) remove divergence from a planted kernel · (4) *(hard)* overlap H2D/compute/D2H with 3 streams and show it in an Nsight Systems timeline.

**Sources.** PMPP ch 2–4 · CUDA Programming Guide (programming model, hardware implementation) · cuda-samples (`cpp/0_Introduction/`, `cpp/1_Utilities/deviceQuery/`; verified at the pinned SHA, which moved samples from `Samples/` to `cpp/`) · handbook GPU fundamentals pages.

## P5.2 — Memory: coalescing, shared memory, bank conflicts

**Objectives.** (1) Global memory transactions: 32-byte sectors, 128-byte lines, coalescing rules. (2) Shared memory, `__syncthreads`, banks and conflicts, and the padding fix. (3) Matrix transpose: naive → coalesced via shared tile → padded (the L2 exit check: ≥80% of copy bandwidth). (4) `__restrict__`, `__ldg`, vectorized `float4` loads.

**Examples.** `01_strided_read.cu` (bandwidth vs stride, a T2 bench) · `02_transpose_ladder.cu` (3 rungs) · `03_bank_conflicts.cu`.

**Exercises.** (1) Predict sectors per request for 4 access patterns, then confirm with `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` · (2) transpose ≥80% of copy · (3) remove the bank conflicts from a planted kernel (ncu shared-memory metric goes to ~0) · (4) *(hard)* 2D convolution with halo tiles.

**Animations (required).** `animations/p5-coalescing.html` (warp lanes → addresses → sectors fetched, contiguous vs strided vs misaligned) · `animations/p5-smem-tiling.html` (tile loads, sync, bank map with and without +1 padding).

**Sources.** PMPP ch 5–6 · Best Practices Guide (memory optimizations) · handbook `gpu-memory.md`.

## P5.3 — Profiling with Nsight Compute and Nsight Systems

**Objectives.** (1) Nsight Systems: timelines, CPU/GPU gaps, NVTX ranges. (2) Nsight Compute: Speed-of-Light, memory workload analysis, warp-state stall reasons, source view, and its roofline chart. (3) On cloud: counter access (`ERR_NVGPUCTRPERM`) via `sudo` or `NVreg_RestrictProfilingToAdminUsers=0`, then pull `.ncu-rep` back for the desktop UI. (4) `ncu --csv` summaries committed next to every benchmark.

**Exercises.** (1) Diagnose 3 planted slow kernels from ncu reports alone (written answers + fix) · (2) NVTX-annotate the P0.5/#0 inference loop and read the nsys timeline · (3) a script that turns `ncu --csv` into the bench table's "% of peak" column.

**Sources.** Nsight Compute / Systems docs · ERR_NVGPUCTRPERM page · GPU MODE lecture on profiling (path in `gpu-mode/lectures` verified at build time).

## P5.4 — Reductions and scans

**Objectives.** (1) Harris's reduction ladder: interleaved → sequential addressing → first add during load → unrolled → warp-level with `__shfl_down_sync` → grid-stride + one atomic. (2) Cooperative groups. (3) Scan: Hillis–Steele vs Blelloch, then decoupled look-back (why CUB is fast). (4) Baseline: `cub::DeviceReduce::Sum` / `cub::DeviceScan` from CCCL `v3.5.0`.

**Examples.** `01_reduce_ladder.cu` · `02_warp_reduce.cu` · `03_scan.cu` · `04_cub_baseline.cu`.

**Exercises.** (1) Each rung passes the harness and reports GB/s · (2) reduction within 10% of CUB (the L3 exit check) · (3) block scan + scan-then-propagate for large arrays · (4) *(hard)* top-k via per-block selection + merge.

**Animation (required).** `animations/p5-reduction.html` (tree in shared memory vs shuffle across lanes, step by step).

**Sources.** Mark Harris, *Optimizing Parallel Reduction in CUDA* · PMPP ch 10–11 · CCCL (CUB) · llm.c `dev/cuda/` (reduction patterns inside `layernorm_forward.cu`).

## P5.5 — Softmax and norms

**Objectives.** (1) Safe softmax (3 passes) → online softmax (2 passes) → single-pass online softmax per row. (2) RMSNorm/LayerNorm: one warp or block per row, `float4` loads, fp32 accumulation for fp16/bf16 data, with tolerances explained. (3) Fusion: residual add + RMSNorm in one pass, and why fusion is a bandwidth win.

**Examples.** `01_softmax_ladder.cu` · `02_rmsnorm.cu` · `03_fused_residual_rmsnorm.cu` · `04_fp16_bf16.cu`.

**Exercises.** (1) Online softmax, reported in GB/s (the L4 exit check) · (2) RMSNorm within X% of copy bandwidth (X set after a measured baseline) · (3) bf16 tolerance test with justification (bf16 has an 8-bit mantissa, so `rtol≈1e-2`; this is derived in the lesson) · (4) *(hard)* fused cross-entropy (log-sum-exp + gather).

**Animation (required).** `animations/p5-online-softmax.html`.

**Sources.** llm.c `dev/cuda/softmax_forward.cu`, `layernorm_forward.cu` (read, annotate ≤15-line excerpts, MIT) · Triton tutorial `02-fused-softmax.py`, `05-layer-norm.py` (for P5.9 comparison) · handbook `kernel-optimization-for-llm-inference.md`.

## P5.6 — The GEMM ladder (rungs 1–7)

**Rungs** (from the prompt; each its own kernel, tested and benchmarked): (1) naive · (2) coalesced · (3) shared-memory tiling · (4) 1D register blocking · (5) 2D register blocking · (6) vectorized loads + transposed A tile · (7) warp tiling + autotuning + double buffering.

**Objectives.** For each rung, predict the speedup from arithmetic intensity and memory traffic, measure it, and explain the change with one ncu metric. Compare against cuBLAS SGEMM at sizes 1024–8192. Exit check (L5): ≥70% of cuBLAS.

**Examples.** `gemm/01_naive.cu` … `gemm/07_warptile.cu`, sharing one runner · `gemm/autotune.py`.

**Exercises.** Implement rungs 3–7 from starters (each has a harness test vs cuBLAS with `rtol=1e-3`, fp32) · *(hard)* rectangular and non-multiple-of-tile shapes with predication.

**Bench.** `N | rung | ms | TFLOP/s | % cuBLAS | % peak`, plus a JSON chart.

**Sources.** siboehm SGEMM_CUDA (kernels `src/kernels/1_naive.cuh` … `12_kernel_double_buffering.cuh`; read and explain only, not vendored) and article · PMPP ch 6 · CUTLASS docs (efficient GEMM overview) · Algorithms for Modern Hardware matmul chapter (CPU parallel to D1).

## P5.7 — Tensor cores: WMMA → `mma.sync` → CuTe (rung 8)

**Objectives.** (1) Fragments and the WMMA API (sm_70+). (2) `mma.sync` PTX with `ldmatrix` and fragment layouts (sm_80+). (3) `cp.async` and multi-stage pipelines. (4) CuTe layouts and tensors from `media/docs/cpp/cute/` (01_layout → 0x_gemm_tutorial) and rewriting the HGEMM in CuTe. (5) INT8 IMMA / `dp4a` quantized matmul. Exit check: tensor-core HGEMM ≥50% of cuBLAS.

**Examples.** `01_wmma_hgemm.cu` · `02_mma_sync_hgemm.cu` · `03_cpasync_pipeline.cu` · `04_cute_hgemm.cu` · `05_int8_gemm.cu`.

**Exercises.** (1) WMMA HGEMM (fp16 in, fp32 accumulate, harness tolerance derived) · (2) `ldmatrix` + `mma.sync` · (3) 2-stage → 3-stage pipeline gain · (4) *(hard)* CuTe HGEMM ≥50% cuBLAS.

**Sources.** CUDA Programming Guide (WMMA, async copy) · PTX ISA (mma) · CUTLASS `v4.8.0` `examples/` + CuTe docs · PMPP (tensor core chapter, 4th ed.).

## P5.8 — FlashAttention v1 → v2 forward

**Objectives.** (1) Naive attention: materialize S = QKᵀ, softmax, PV, and count its HBM traffic. (2) FA-1: tile over K/V, online softmax, never write S. (3) FA-2: parallelize over the sequence, a better work split, and fewer non-matmul FLOPs. (4) Causal masking with tile skipping. (5) Exit check (L6): FA-2 forward ≥5× naive at seq 4k (harness, fp16).

**Examples.** `01_naive_attention.cu` · `02_fa1.cu` · `03_fa2.cu` · `04_causal.cu` · `05_vs_flash_attn.py` (compare with the pinned `flash-attn v2.8.3.post1` and PyTorch SDPA).

**Exercises.** (1) Naive attention vs a PyTorch reference · (2) FA-1 from the minimal teaching version's structure (tspeterkim `flash.cu`, read and annotated, not copied) · (3) FA-2 forward + causal · (4) *(hard)* decode attention (q_len=1) over a paged KV layout, which feeds P6.3.

**Animation (required).** `animations/p5-flashattention.html` (Q block row stationary, K/V blocks streaming, running max/sum updated per tile).

**Sources.** FA-1 (2205.14135), FA-2 (2307.08691), FA-3 (2407.08608, overview only) · Dao-AILab/flash-attention · tspeterkim/flash-attention-minimal · handbook `flashattention.md` · LeetGPU L6 problems.

## P5.9 — Triton

**Objectives.** (1) The block-level programming model, `tl.load`/`tl.store` masks, `tl.dot`, autotune configs. (2) Rewrite softmax, RMSNorm, GEMM and FA-2 forward in Triton and compare them with your CUDA versions. (3) Interpreter mode (`TRITON_INTERPRET=1`) for T0 correctness tests in CI.

**Examples.** Triton `v3.8.0` tutorials `01`, `02`, `03`, `05`, `06` (run and annotated, linked not copied) · `ours/softmax.py`, `ours/rmsnorm.py`, `ours/matmul.py`, `ours/fa2.py`.

**Exercises.** (1–4) Each rewrite passes pytest on T0 (interpreter) and T2 (real), with a CUDA vs Triton bench table · (5) *(hard)* Triton Puzzles subset (linked).

**Sources.** triton-lang/triton `python/tutorials/` · Triton Puzzles · GPU MODE Triton lecture.

## P5.10 — Wiring a custom op into PyTorch/vLLM (+ #12)

**Objectives.** (1) PyTorch custom ops: `torch.library` + a C++/CUDA extension, `torch.compile` compatibility (fake/meta kernels), and opcheck tests. (2) Where vLLM dispatches its kernels (custom ops registry, `csrc/`) and how to swap an implementation behind a flag in a fork pinned to `v0.31.0`. (3) **#12:** replace one hot path (RMSNorm+residual or a sampling kernel) with your Triton/CUDA kernel. Before/after is measured end-to-end with #4 and traced with #13.

**Exercises.** (1) Register an RMSNorm op and pass `torch.library.opcheck` · (2) make it `torch.compile`-safe · (3) **#12** end-to-end bench · (4) *(hard)* upstream-quality PR description for the change (not submitted unless you choose to).

**Sources.** PyTorch custom ops docs (UNVERIFIED) · vLLM `csrc/`, `vllm/_custom_ops.py` (both verified at the pinned SHA) · SGLang `python/sglang/kernels/`.
