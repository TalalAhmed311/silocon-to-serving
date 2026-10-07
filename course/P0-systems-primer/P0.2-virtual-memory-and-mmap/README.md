# P0.2 — Virtual memory and mmap

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f). Linux gives the full fault counters; macOS works for everything except `MADV_HUGEPAGE` and dropping the page cache |
| **Time** | ≈30 min reading + ≈6 h hands-on |
| **Prerequisites** | P0.1 (pointers, strides, RAII, CMake) |
| **You will build** | the safetensors mmap loader that P0.5's engine uses |

## Learning objectives

1. Trace a virtual address through a 4-level page table and the TLB, and explain what a page fault costs.
2. Explain demand paging and the page cache, and measure the cost of touching an `mmap`'d file for the first time vs a second time.
3. Explain huge pages and **TLB reach**, and measure `MADV_HUGEPAGE` on Linux.
4. Explain pageable vs pinned memory and why DMA engines (and so `cudaMemcpyAsync`) need pinned pages.
5. Parse a safetensors header and map its tensors with zero copies.

## Why this matters

When vLLM starts, the biggest part of its cold-start time is usually moving weights:

```
object store → disk → page cache → (pinned) host memory → GPU HBM
```

Every arrow in that chain is a virtual-memory concept from this module. The CPU engine in P0.5 loads a model with **one `mmap` call and no copies**. P3.7 (weight delivery) and P5.1 (pinned vs pageable `cudaMemcpy`) build on exactly these ideas.

---

## 1. Every address is virtual

Your process never sees physical RAM addresses. Every load and store uses a **virtual address (VA)**, which the CPU's MMU translates to a **physical address (PA)** using **page tables** that the OS maintains per process. Memory is managed in **pages**: 4 KiB by default on x86-64 and Linux/arm64, and 16 KiB on Apple silicon.

On x86-64 with 4-level paging, a 48-bit VA splits like this:

```
| 47..39 | 38..30 | 29..21 | 20..12 | 11..0  |
|  PML4  |  PDPT  |   PD   |   PT   | offset |   9 + 9 + 9 + 9 index bits, 12 offset bits (4 KiB)
```

Each 9-bit index selects one of 512 entries in a table, and each table is itself one 4 KiB page. A full translation is therefore **four dependent memory reads** before your actual load can happen. That is why the CPU caches translations.

## 2. The TLB

The **Translation Lookaside Buffer** is a small cache of recent VA-page → PA-frame translations. A TLB hit costs about nothing. A miss triggers a **page walk**: up to 4 reads, which are often served from the L1/L2 caches, but still add tens of cycles of latency.

**TLB reach** = (number of entries) × (page size). It is the amount of memory you can touch randomly without missing in the TLB:

| Page size | Example: 1536 L2-TLB entries | Reach |
|---|---|---|
| 4 KiB | 1536 × 4 KiB | 6 MiB |
| 2 MiB (huge page) | 1536 × 2 MiB | 3 GiB |

(The 1536-entry figure is an example for the arithmetic. Your CPU's real count is in its optimization manual, and exercise 1 asks you to look yours up.) A 16 GB weight file accessed through 4 KiB pages blows far past TLB reach. Huge pages cut the number of translations 512×. This is why GPU drivers and large-memory servers use huge pages.

## 3. Page faults and demand paging

`mmap(file)` does **not** read the file. It only creates page-table entries marked "not present". The first touch of each page causes a **page fault**: the CPU traps into the kernel, which:

- **minor fault:** finds the file's page already in the **page cache** (RAM the kernel uses to cache file data) and just maps it. This costs about a microsecond.
- **major fault:** must read the page from disk first. This costs an SSD read, tens to hundreds of microseconds.

The second pass over the same file causes no faults at all, because the pages are mapped. `examples/02_first_touch.c` measures all three cases with `getrusage()` fault counters:

- cold: after dropping the page cache (Linux, root only)
- warm cache, new mapping
- already mapped

> **Predict first.** You `mmap` a 1 GiB file whose pages are already in the page cache, and read one byte per 4 KiB page. How many minor faults do you expect? (1 GiB ÷ 4 KiB = 262,144, unless the kernel's fault-around maps several neighbouring pages per fault. Linux does this by default for file mappings.) Write down a number, then compare with what `02_first_touch` prints. The gap teaches you about *fault-around*.

## 4. `mmap` vs `read`

| | `read()` into a buffer | `mmap()` |
|---|---|---|
| copies | page cache → your buffer (one extra copy) | none: you read the page-cache pages directly |
| memory | your buffer + the page cache | only the page cache (shared by every process mapping the file) |
| when data arrives | at the `read` call | lazily, on first touch (or `MADV_WILLNEED` / `MAP_POPULATE`) |

Two processes that map the same model file share **one** physical copy. That is how llama.cpp lets several processes load one model without duplicating it in RAM.

## 5. Pinned vs pageable memory, and DMA

A **DMA engine**, such as the GPU's copy engine or an NVMe controller, reads physical memory directly, without the CPU. It needs the physical pages to stay put for the whole transfer. Ordinary (**pageable**) memory can be swapped out or migrated by the OS at any time. **Pinned (page-locked)** memory cannot.

So when you `cudaMemcpy` from pageable memory, the driver first copies your data into an internal pinned **staging buffer**, then DMAs from there. That is an extra memcpy. Allocating with `cudaMallocHost`, or pinning with `cudaHostRegister`, removes it, and is required for truly asynchronous copies. On a laptop without a GPU, `mlock()` is the closest analogue. `examples/04_mlock_pinned.c` shows the mechanics. The real GPU measurement is in P5.1.

## 6. safetensors: a format designed for mmap

A `.safetensors` file is:

```
[8 bytes: little-endian u64 N] [N bytes: UTF-8 JSON header] [raw tensor bytes ...]
```

The header maps each tensor name to `{"dtype": "F32", "shape": [rows, cols], "data_offsets": [begin, end]}`, with offsets relative to the start of the data section. There is an optional `"__metadata__"` entry. Because the data is raw and little-endian, a loader can `mmap` the file and hand out pointers `base + 8 + N + begin` **without copying or parsing a single weight**. (Format spec: the README of `huggingface/safetensors@e246a256`.)

`examples/05_safetensors_mmap.cpp` is a ~150-line loader. In exercise 3 you harden it with bounds checks, dtype validation and alignment checks, and it becomes the loader for P0.5.

---

## Walkthrough

```bash
cd course/P0-systems-primer/P0.2-virtual-memory-and-mmap
uv run python examples/01_page_walk.py                  # simulate VA → PA with a TLB
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release && cmake --build build/examples -j
./build/examples/02_first_touch /tmp/s2s-1g.bin 1024     # creates a 1 GiB file if missing
./build/examples/03_hugepages                            # Linux only for the THP variant
./build/examples/04_mlock_pinned
uv run python examples/make_tiny_safetensors.py /tmp/tiny.safetensors
./build/examples/05_safetensors_mmap /tmp/tiny.safetensors
```

## What you should see

- `01_page_walk.py` prints each VA split into its 4 indices + offset, then a TLB hit/miss trace whose hit rate grows with locality. The output is deterministic (seeded); the first lines are listed in the script's header.
- `02_first_touch` prints `minor/major faults` and ms for each pass. **TODO(run): paste your table.** Expected shape:
  - the first pass over a new mapping has many minor faults (fewer than the page count if fault-around is active)
  - the second pass has ~0 faults and is several times faster
  - after `echo 3 > /proc/sys/vm/drop_caches` (root), the first pass also shows major faults and runs at disk speed
- `05_safetensors_mmap` lists every tensor with dtype, shape and offset, plus a checksum. Its sums must match `make_tiny_safetensors.py`'s to ~6 significant digits (NumPy uses pairwise summation, the C++ loop sums sequentially).

## Exercises

```bash
uv run pytest exercises -q                                      # Python exercises (1, 2)
cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex --output-on-failure  # C++ (3, 4)
# reference solutions: S2S_SOLUTIONS=1 uv run pytest exercises ; cmake ... -DS2S_USE_SOLUTIONS=ON
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [TLB reach calculator](exercises/01-tlb-reach/README.md) | easy | pytest |
| 2 | [Predict page faults for an access pattern](exercises/02-predict-faults/README.md) | easy | pytest (simulated, deterministic) |
| 3 | [Harden the safetensors loader](exercises/03-safetensors-loader/README.md) | medium | ctest against files made by the Python `safetensors` writer + 6 malformed files |
| 4 | [Prefetch with `MADV_WILLNEED`](exercises/04-prefetch/README.md) | hard | ctest (correctness) + bench table (speed) |

## Benchmark

`bench/mmap_bench.c` measures cold vs warm load of a weight file and prints:

```
| file size | variant | median ms | p90 ms | GB/s | % of peak |
```

Variants: `read()` into a buffer, `mmap` + touch, and `mmap` + `MAP_POPULATE`. "Peak" is the warm `read()` row, i.e. memory-to-memory, measured in the same run. Dropping the page cache needs root. Without root, the bench says so and reports the warm numbers only.

## Common mistakes

- **Timing `mmap()` itself.** It returns almost instantly, because the cost comes later, at the first touch. Always time the *touch*.
- **Assuming 4 KiB pages.** Apple silicon uses 16 KiB pages. Use `sysconf(_SC_PAGESIZE)`.
- **Forgetting `munmap` / `close`.** Use RAII (P0.1). `MappedFile` in the examples owns both.
- **Trusting a file header.** A malicious or truncated `.safetensors` file can claim offsets past the end of the file. Bounds-check *everything* (exercise 3).
- **Unaligned typed access.** Casting `base + offset` to `float*` requires `offset % 4 == 0`. safetensors writers align data, but check anyway.

## Go deeper

- OSTEP: *Address Translation*, *Paging: Faster Translations (TLBs)*, *Paging: Smaller Tables*, *Beyond Physical Memory* (Part I, Virtualization).
- CS:APP ch. 9 "Virtual Memory" (§9.6 address translation, §9.8 memory mapping).
- `huggingface/safetensors@e246a256` README (format) and `safetensors/src/tensor.rs` (the reference parser's validation rules).
- llama2.c `run.c`, `memory_map_weights()` and `read_checkpoint()`, for `mmap` of a model file in ≈30 lines.

## Go down when…

You're at the bottom. **Next:** [P0.3 Threads, atomics, caches](../P0.3-threads-atomics-caches/README.md). **Comes back in:** P0.5 (loader), P3.7 (weight delivery), P5.1 (pinned memory on a GPU).

## Animation

[`animations/p0-vm-page-walk.html`](../../../animations/p0-vm-page-walk.html) shows VA bits → page-table levels → TLB hit/miss → physical frame, then a "first touch" mode where a page fault pulls a file page from the page cache into a frame.
