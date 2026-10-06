# P0 — Systems primer: C/C++, OS, memory, SIMD

**Weeks 1–3 · Lane A ≈ 33 h · Tier T0 (laptop, Linux/macOS, x86 or ARM) · Lane B: L1 runs alongside (3–4 problems/week)**

**Why this phase comes first:** every later layer, from vLLM's block manager to a CUDA kernel, depends on four ideas: memory layout, virtual memory, threads, and caches/SIMD. P0 teaches them on the CPU, where you can see and measure everything. It ends with a working CPU inference engine (#0 v0), which later becomes the GPU engine (#0 v1, P6.7).

| Module | Time (read + hands-on) | Project / drill | Animations |
|---|---|---|---|
| P0.1 C/C++ for systems | 0.5 h + 6 h | — | memory-layout explorer (struct padding, row- vs column-major) |
| P0.2 Virtual memory and mmap | 0.5 h + 6 h | (loader for #0 v0) | **VA → page table → TLB → physical page; page fault on first touch of an mmap'd file** |
| P0.3 Threads, atomics, caches | 0.6 h + 7 h | **D2 KV block allocator** | **false sharing between two cores on one cache line** |
| P0.4 SIMD and roofline | 0.6 h + 7 h | **D1 SIMD SGEMM + roofline** | **SIMD lanes: scalar vs 8-wide AVX2 add, misalignment cost**; **roofline (drag arithmetic intensity)** |
| P0.5 Project: Tiny Inference Engine (CPU) | 0.4 h + 6 h | **#0 v0** | reuses P0.2 + P0.4 animations |

---

## P0.1 — C/C++ for systems

**Objectives.** After this module you can:

1. Predict `sizeof` and field offsets of any struct (alignment and padding) and explain how they affect cache use.
2. Use pointer arithmetic over flat row-major and column-major buffers, plus strides, which is the same idea as tensor strides.
3. Write RAII owners with move semantics. You know when a copy happens, and you can prove it with a counter.
4. Write a small function template and a `constexpr` helper, and use `std::span`/`std::unique_ptr` correctly.
5. Build a multi-target project with modern CMake (targets, `target_compile_options`, `-O3 -march=native`, sanitizers, CTest).

**Examples** (`examples/`, smallest first):

- `01_layout.c`: struct padding, `offsetof`, `alignas`.
- `02_strides.c`: one buffer viewed as row-major, column-major and transposed through strides.
- `03_raii_buffer.cpp`: an aligned buffer owner (`std::aligned_alloc`). Copy is deleted and move is counted.
- `04_move_vs_copy.cpp`: the same pipeline with copies vs moves, timed. Naive vs optimized.
- `05_templates.cpp`: a `Matrix<T>` view template, a `constexpr` tile-size helper, and C++20 concepts.
- `CMakeLists.txt`: Release/Debug/ASan presets, CTest registration.

**Exercises** (`exercises/`, easy → hard; each has a starter, a `solutions/` reference and a ctest):

1. Reorder fields to shrink a struct. The test checks `sizeof`.
2. Implement `strided_copy` (tensor `.contiguous()` for 2D). The test compares it with a naive loop.
3. Write the `Tensor` RAII class: an aligned owner plus shape/stride and move-only semantics. Tests cover move counts and ASan cleanliness.
4. Implement a generic `transpose<T>` with a cache-blocking template parameter. Tested for correctness across T ∈ {float, int8_t}.
5. *(hard)* Find and fix three planted memory bugs using ASan/UBSan output. Tests run under the sanitizers.

**Bench.** `bench/transpose_bench.cpp` measures naive vs blocked transpose and prints a table: `size | time | GB/s | % of STREAM copy`. It also saves JSON. The peak comes from a STREAM-style copy run in the same script, so % of peak is measured, not assumed.

**Predict first.** For a 4096×4096 float transpose, predict the naive version's GB/s relative to a plain copy, from cache-line size and stride. Then measure.

**Sources.** CS:APP ch 6 (cache-friendly code) · learncpp (move semantics, templates) · *C++ Concurrency in Action* ch 1 (setup only) · llama2.c `run.c` (its `Config`/`TransformerWeights` structs and pointer slicing are a worked layout example).

**Go down when…** none (this is the bottom of the CPU path). **Go up to:** P0.2.

---

## P0.2 — Virtual memory and mmap

**Objectives.**

1. Trace a virtual address through a 4-level page table and the TLB, and explain what a page fault costs.
2. Explain demand paging and the page cache. Measure the first-touch vs warm cost of an `mmap`'d file.
3. Explain huge pages and TLB reach, and measure the effect of `MADV_HUGEPAGE` (Linux).
4. Explain pageable vs pinned (page-locked) memory and why DMA engines need pinned pages. This is the CPU half of `cudaMallocHost`, finished in P5.
5. Parse a safetensors header and map tensors zero-copy.

**Examples.**

- `01_page_walk.py`: simulates a page walk with a TLB and prints hits and misses.
- `02_first_touch.c`: `mmap` a 1 GB file, time first vs second pass, and read minor/major faults from `getrusage`.
- `03_hugepages.c`: random access over a large buffer with and without huge pages (Linux).
- `04_mlock_pinned.c`: `mlock`'d vs pageable memcpy timing, as a stand-in for pinned memory on a laptop.
- `05_safetensors_mmap.cpp`: header JSON parse, offsets, and `mmap` views. Tested against the Python `safetensors` loader.

**Exercises.**

1. Compute TLB reach for 4 KB vs 2 MB pages (pytest on a calculator function).
2. Count page faults for a stride pattern and predict the count before running.
3. Write the safetensors loader for #0 v0: dtype table, alignment checks, bounds checks against the file size. Tested against the official Python loader on a tiny model.
4. *(hard)* Add a `MADV_WILLNEED`/`readahead` prefetch path and show the cold-load improvement in a bench table.

**Bench.** `bench/mmap_bench.c` compares cold vs warm load of a weight file: `file size | cold ms | warm ms | GB/s`. Dropping the page cache needs root on Linux. If root isn't available, the script says so and prints both numbers it *can* measure.

**Animation (required).** `animations/p0-vm-page-walk.html`: VA bits → page-table levels → TLB hit/miss → physical frame, and a "first touch" mode where a page fault fills a frame from the page cache.

**Sources.** OSTEP Virtualization chapters (address translation, TLBs, paging, swapping) · CS:APP ch 9 · safetensors repo (format spec in its README at the pinned SHA) · llama2.c `run.c` (`memory_map_weights`).

**Go down when…** none. **Go up to:** P0.5 (loader), P3.7 (weight delivery reuses this exactly).

---

## P0.3 — Threads, atomics, caches

**Objectives.**

1. Contrast processes and threads (address space, cost to create, sharing).
2. Use `std::thread`, `std::mutex`, `std::condition_variable` and `std::atomic`, with memory orders explained for acquire/release only.
3. Detect and fix false sharing with `alignas(64)` / `std::hardware_destructive_interference_size`.
4. Build a fixed-size thread pool with a work queue.
5. Explain NUMA and measure local vs remote memory on a multi-socket machine. On a laptop this is explained, not measured, and is marked T3-optional.
6. Read a `perf stat` / `perf record` report and a flame graph.

**Examples.**

- `01_proc_vs_thread.c`: creation cost.
- `02_false_sharing.cpp`: packed vs padded counters (naive vs optimized).
- `03_atomics.cpp`: a mutex counter vs an atomic counter vs per-thread counters summed at the end.
- `04_thread_pool.cpp`: the pool used by #0 v0 matmul.
- `05_perf_walkthrough.md`: `perf stat -e cache-misses,…` on example 02. macOS learners get Instruments notes.

**Exercises.**

1. Fix the planted data race. The test runs under TSan.
2. Pad the counters, then show the speedup in the bench table.
3. Write the `ThreadPool::parallel_for(n, fn)`. The test is a parallel sum compared with a serial reference.
4. **D2: KV block allocator.** Fixed-size blocks, free list, ref counts, `fork()` (share blocks), `free()` and a stress test with random alloc/fork/free across threads. Tests check invariants: no double free, refcount never negative, and all blocks free at the end. This is the precursor to P6.3 PagedAttention.
5. *(hard)* Make D2 lock-free for the single-producer/single-consumer case and prove it under TSan.

**Bench.** `bench/false_sharing_bench.cpp` prints `threads | packed ns/op | padded ns/op | speedup`. `bench/allocator_bench.cpp` prints allocations/s against the thread count.

**Animation (required).** `animations/p0-false-sharing.html`: two cores, one 64-byte line, MESI state ping-pong per write, then the padded version.

**Sources.** OSTEP Concurrency chapters · *C++ Concurrency in Action* ch 2–5, 9 (thread pools) · CS:APP ch 6 · Brendan Gregg perf and flame-graph pages.

**Go up to:** P2.3 (prefix caching = shared blocks), P6.3.

---

## P0.4 — SIMD and roofline

**Objectives.**

1. Write AVX2 (x86) or NEON (ARM) intrinsics for add, FMA and horizontal sum. Use the right header and compile flags.
2. Handle tails and alignment, and measure what misalignment actually costs on your CPU.
3. Compute arithmetic intensity (FLOP/byte) for dot product, matvec and matmul, and place each on a roofline.
4. Measure your machine's peak FLOP/s (an FMA microbenchmark) and bandwidth (STREAM-style), and draw your own roofline.
5. Climb the SGEMM ladder: naive → loop reorder → tiled → SIMD → multithreaded (D1).
6. Write an int8 dot product with int32 accumulation (AVX2 `maddubs` or NEON `sdot`) and compare it with llama.cpp's `q8_0` dot product structure.

**Examples.**

- `01_scalar_vs_simd_add.cpp`: same loop scalar, auto-vectorized (check `-fopt-info-vec` / `-Rpass=loop-vectorize`) and intrinsics.
- `02_fma_peak.cpp`: measures peak GFLOP/s, with enough independent accumulators to hide FMA latency.
- `03_stream.cpp`: copy/scale/add/triad bandwidth.
- `04_roofline.py`: reads both JSONs and plots the roofline plus measured kernels (matplotlib, saves a PNG).
- `05_int8_dot.cpp`: scalar vs SIMD int8 dot, with an annotated ≤15-line excerpt pointer into llama.cpp `ggml/src/ggml-cpu/arch/x86/quants.c` (MIT).

**Exercises.**

1. Vectorize a SAXPY with tail handling (ctest vs scalar).
2. Horizontal sum in under 6 instructions (ctest).
3. **D1: SIMD SGEMM.** Five rungs in one file behind a flag. Each rung is tested against a NumPy-generated reference at fp32 `rtol=1e-4`.
4. Multithread D1 with the P0.3 pool. Report the scaling efficiency.
5. *(hard)* Int8 GEMV with per-block scales (a `q8_0`-like format of your own design). Test against fp32 within the stated quantization error.

**Bench.** `bench/sgemm_bench.py` drives the D1 binary and prints `N | rung | ms | GFLOP/s | % of measured peak`. It saves JSON for the roofline plot.

**Predict first.** From your measured peak and bandwidth, predict the ridge point. Then predict whether N=64 and N=2048 SGEMM are memory- or compute-bound, and check with the roofline.

**Animations (required).** `animations/p0-simd-lanes.html` (scalar vs 8 lanes, aligned vs a load split across cache lines) and `animations/roofline.html` (drag the arithmetic intensity and watch the kernel cross the ridge; CPU and GPU presets, GPU presets filled from P1.4's verified numbers).

**Sources.** Algorithms for Modern Hardware: SIMD and matmul chapters · Intel Intrinsics Guide / Arm intrinsics · Agner Fog manuals (instruction tables) · CS:APP ch 5 · llama.cpp `ggml/src/ggml-cpu/` and block formats in `ggml/src/ggml-common.h`.

**Go up to:** P1.4 (the GPU roofline uses the same model), P5.6 (GPU GEMM ladder mirrors D1).

---

## P0.5 — Project: Tiny Inference Engine (CPU), #0 v0

**Objectives.**

1. Implement a Llama-architecture forward pass in C++: embedding, RMSNorm, RoPE, attention with a KV cache, SwiGLU MLP and the LM head.
2. Load weights from safetensors through your P0.2 mmap loader, with no copies.
3. Use D1's SIMD matmul and P0.3's thread pool for every matvec.
4. Implement temperature, top-k and top-p sampling with a seeded RNG.
5. Measure tokens/s and compare it with a bandwidth-bound prediction.

**Model.** A tiny Llama-architecture checkpoint, small enough for a laptop and CI. The candidate is the llama2.c TinyStories models re-exported to safetensors. The exact model and SHA are pinned at build time and listed in SOURCES.md.

**Examples.** `01_forward_numpy.py` (the reference forward pass that every C++ test compares against), `02_engine/` (C++ CMake project), `03_sampling.cpp`.

**Exercises.**

1. RMSNorm and RoPE in C++ vs NumPy, fp32 `atol=1e-5`.
2. Single-layer attention with a KV cache: logits match NumPy at every position.
3. Full forward pass: greedy decode of 64 tokens is token-identical to the NumPy reference.
4. Top-p sampling: distribution test (χ² vs expected frequencies over 10⁵ draws with a fixed seed).
5. *(hard)* Add `runq`-style int8 weights using the P0.4 format. Report perplexity change on a fixed text and the speedup.

**Bench.** `bench/engine_bench.py` prints `model | threads | prompt toks | prefill tok/s | decode tok/s | predicted decode ceiling`. The ceiling is computed as measured bandwidth (P0.4 STREAM) ÷ bytes of weights read per token.

**Predict first (the worked prediction).** Decode tok/s ≈ memory bandwidth ÷ model bytes. You use your own measured STREAM bandwidth and the model's actual file size, then explain the gap: thread overhead, KV reads, and the compute-bound LM head at small sizes.

**Sources.** llama2.c `run.c` (structure; we do not copy it) · HF `modeling_llama.py` (RoPE convention check) · handbook `inference-parameters.md` (TopPvsTopK interactive, linked).

**Go down when…** a kernel is slow: P0.4 · loader bugs: P0.2. **Go up to:** P1.1 (same forward pass, now as arithmetic), P6.7 (#0 v1).
