# P5.2: Memory (coalescing, shared memory, bank conflicts)

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) (sm_75+; `g4dn.xlarge` in [aws.md](aws.md)) |
| **Time** | ≈30 min reading + ≈10 h hands-on |
| **Prerequisites** | P5.1, P0.4 (bandwidth rooflines) |
| **Reading** | PMPP ch. 5–6 · CUDA Best Practices Guide: *Memory Optimizations* · Modular handbook: *GPU memory* |

## Learning objectives

1. Predict how many **32-byte sectors** a warp's load touches, and turn that into effective bandwidth.
2. Use **shared memory** to turn strided global access into coalesced access, with `__syncthreads()` in the right places.
3. Explain **bank conflicts** and remove them with padding or a different mapping.
4. Reach the **L2 exit check**: transpose at ≥ 80% of measured copy bandwidth.
5. Use `__restrict__`, the read-only path (`__ldg`) and `float4` vector loads, and know when each helps.

---

## 1. Global memory: sectors

The memory system moves **32-byte sectors**, grouped into 128-byte cache lines. When a warp executes a load, the hardware works out which sectors its 32 addresses touch and fetches each one once.

| warp access (4-byte floats) | sectors | useful / fetched |
|---|---|---|
| 32 consecutive, aligned | 4 | 100% |
| 32 consecutive, offset by 1 float | 5 | 80% |
| stride 2 | 8 | 50% |
| stride ≥ 8 | 32 | 12.5% |

Effective bandwidth = peak × useful/fetched, so rule 1 is **adjacent threads touch adjacent addresses**. The animation [`p5-coalescing.html`](../../../animations/p5-coalescing.html) draws lanes → addresses → sectors for each case. `01_strided_read.cu` measures them.

## 2. Shared memory and banks

Shared memory is on-chip SRAM, per block, explicitly managed. It's divided into **32 banks of 4 bytes**: word `w` lives in bank `w mod 32`. A warp's 32 accesses complete in one pass if they hit distinct banks, or the same word (broadcast). **k distinct words in one bank cost k passes.**

A `[32][32]` float tile read by column (lane i reads `tile[i][k]`) puts all 32 lanes in bank k: a 32-way conflict. Declaring `[32][33]` shifts each row by one bank, so `tile[i][k]` lands in bank `(i + k) mod 32`, all distinct. The animation [`p5-smem-tiling.html`](../../../animations/p5-smem-tiling.html) shows the bank map with and without padding.

## 3. Transpose: the ladder

| rung | reads | writes | limiter |
|---|---|---|---|
| 1 naive | coalesced | stride = rows → 32 sectors per request | write sectors |
| 2 smem tile | coalesced | coalesced (written from the tile, column-wise) | 32-way bank conflicts on the tile read |
| 3 padded tile | coalesced | coalesced | ≈ copy bandwidth |

Kernels: `platform/kernels/include/d4/transpose.cuh` (D4). The exercise has you write rung 3 yourself.

## 4. Smaller levers

- `__restrict__` tells the compiler pointers don't alias, so it can keep values in registers and use the read-only path.
- `__ldg(p)` reads explicitly through the read-only/texture path. It's useful for gathered, reused data (SpMV's `x`, L3).
- `float4` loads are 16 bytes per instruction: fewer instructions for the same bytes. Valid only with 16-byte alignment.

> **Predict first.** A naive transpose of an 8192 × 8192 float matrix: reads are 100% efficient, and writes fetch 32 sectors per request with 4 useful bytes each. If the copy kernel runs at B GB/s, roughly what GB/s do you expect from the naive transpose? (Hint: L2 absorbs some write traffic, so your estimate will be pessimistic. Say by how much once you measure.)

---

## Walkthrough

See [aws.md](aws.md): run the three examples, then the exercises with `--bench`, then ncu for the sector and bank-conflict counters.

## What you should see

- `01`: useful GB/s halves per stride doubling until stride 8, then stays flat. The misaligned case costs a little (5 vs 4 sectors).
- `02`: naive < smem < padded ≈ copy. The padded rung passes the exit check.
- `03`: column/no-pad several times slower than row. Padding restores it.

Numbers: `TODO(run-on: g4dn.xlarge)`.

## Bench

| rung | GB/s | % of copy |
|---|---|---|
| 1 naive | | `TODO(run-on: g4dn.xlarge)` |
| 2 smem tile | | |
| 3 smem tile + pad | | (exit: ≥ 80%) |

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Predict sectors per request for 4 patterns, confirm with ncu](exercises/01-sectors.md) | your table vs ncu |
| 2 | [Transpose ≥ 80% of copy (L2 exit check)](exercises/02-transpose/README.md) | `p5.2_02-transpose` (exact) + `--bench` |
| 3 | [Remove the bank conflicts from a planted kernel](exercises/03-bank-conflicts/README.md) | `p5.2_03-bank-conflicts` + ncu counter ~0 |
| 4 | *(hard)* [2D convolution with halo tiles](exercises/04-conv2d-halo/README.md) | `p5.2_04-conv2d-halo` |

## Common mistakes

- Forgetting `__syncthreads()` between filling and reading a shared tile. The results are then *usually* right, which is the worst case.
- `__syncthreads()` inside a branch some threads skip: a deadlock or undefined behavior.
- Measuring bandwidth on a buffer that fits in L2 (T4: 4 MB, L4: 48 MB; UNVERIFIED, so check deviceQuery). Use buffers ≫ L2.
- Padding a tile that's only ever read row-wise: it costs shared memory and fixes nothing.

## Go deeper

- PMPP ch. 5–6. CUDA Best Practices Guide (coalescing, shared memory).
- Modular handbook: MemoryCoalescingVisualizer (linked; ours adds sector counts).
- NVIDIA's "efficient matrix transpose" blog post (classic; UNVERIFIED URL).

**Next:** [P5.3 Profiling with Nsight Compute and Systems](../P5.3-profiling-with-nsight/README.md).
