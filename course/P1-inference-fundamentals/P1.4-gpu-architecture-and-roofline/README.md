# P1.4 — GPU architecture and the roofline

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for everything except example 03 / exercise 4 (![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange)) |
| **Time** | ≈35 min reading + ≈6 h hands-on |
| **Prerequisites** | P0.4 (roofline on a CPU), P1.1–P1.2 |
| **You will build** | `gpu_specs.yaml` (verified by you), GPU rooflines, the "% of peak" helper every later bench uses |

## Learning objectives

1. Describe the GPU hierarchy: GPCs → SMs → warp schedulers and tensor cores, plus registers, shared memory/L1, L2 and HBM.
2. Read an NVIDIA whitepaper or datasheet and pull out the **dense** peak FLOP/s per precision, the memory bandwidth, L2 and SM count. Spot the 2:4-sparsity "×2" in headline numbers.
3. Build rooflines for the GPUs in the course's AWS table and place decode, prefill and attention on them.
4. Turn a roofline into a **tokens/s prediction**, then explain why a real server misses it.

## Why this matters

The CPU roofline from P0.4 carries straight over, with bigger numbers and a much higher ridge. Knowing that an L4 has ~300 GB/s and ~120 dense fp16 TFLOP/s tells you, before you run anything, that:

- batch-1 decode of an 8B fp16 model can't beat ~19 tokens/s
- you need ~400 concurrent tokens of work per weight byte to become compute-bound

> **Every GPU number in this lesson is UNVERIFIED.** The numbers come from `gpu_specs.yaml`, which the build environment could not check against NVIDIA's documents. Exercise 1 is to verify them: open each datasheet or whitepaper, confirm or fix the values, and record where you found each one. Until then, read every number below as "approximately, per the author's memory of the datasheet".

---

## 1. The machine

```
GPU ── several GPCs ── many SMs (Streaming Multiprocessors) ── each SM:
        4 warp schedulers (sub-partitions), each issuing 1 warp-instruction/clock
        CUDA cores (FP32/INT32 ALUs), Tensor Cores (matrix-multiply-accumulate units), SFUs (exp, sin…)
        register file (~64K 32-bit registers), L1 cache / shared memory (a configurable split, ~100–256 KB)
     ── L2 cache (shared by all SMs, tens of MB on Ada/Hopper)
     ── HBM or GDDR (tens of GB, the "memory bandwidth" number)
```

- A **warp** is 32 threads executing in lockstep, the GPU's SIMD unit (P0.4's 8 AVX lanes, but 32 wide and with many warps per SM).
- An SM hides memory latency by **switching between resident warps** each cycle, not with big caches or out-of-order execution. That is why *occupancy* (resident warps per SM) matters (P5.1).
- **Tensor cores** do small matrix multiply-accumulates (for example 16×8×16 fp16 → fp32) in one instruction. They provide almost all of the headline FLOP/s. Plain FP32 CUDA-core throughput is 4–15× lower (compare `fp32_tflops` with `fp16_dense_tflops` in the YAML).

The Modular handbook's GPU fundamentals pages have interactive versions of this picture: SMFloorplan, WarpSchedulerVisualizer and MemoryHierarchyExplorer.

## 2. Reading a datasheet without being fooled

| Trap | What to do |
|---|---|
| "FP16 Tensor: 242 TFLOPS*". The asterisk means *with sparsity* | Divide by 2 for dense. Real models are dense unless you prune them 2:4 |
| FP16 vs "FP16 with FP32 accumulate" | Some consumer cards halve the throughput with FP32 accumulate. Data-center cards usually don't. Check the footnote |
| Boost vs sustained clocks | Peak assumes the boost clock. Sustained clocks under a power cap can be lower: measure (exercise 4) |
| "Memory bandwidth" | It is the theoretical peak. A good copy kernel reaches ~80–90% of it |
| Same name, different SKU | A100 40 GB vs 80 GB, PCIe vs SXM, H100 PCIe vs SXM: different bandwidth and power. Note the exact SKU |

## 3. GPU rooflines

`ridge = peak FLOP/s ÷ bandwidth`. From `gpu_specs.yaml` (dense fp16 tensor):

| GPU | fp16 dense TFLOP/s | GB/s | ridge (FLOP/byte) |
|---|---|---|---|
| T4 | 65 | 320 | ≈ 203 |
| L4 | 121 | 300 | ≈ 403 |
| L40S | 362 | 864 | ≈ 419 |
| A100 80GB SXM | 312 | 2039 | ≈ 153 |
| H100 SXM | 989 | 3350 | ≈ 295 |

(The table is recomputed by `examples/02_gpu_roofline.py` from the YAML, so fixing the YAML fixes the table.)

The ridges sit in the **hundreds**, far right of a CPU's single digits. Decode at batch 1 (I ≈ 1) runs at **< 1%** of tensor peak on all of them. A GPU is a bandwidth machine for decode and a FLOP machine for prefill.

## 4. The worked prediction

> **Predict first.** Llama-3-8B, fp16 weights (≈ 16.06 GB, from P1.1), on an **L4** (≈ 300 GB/s, UNVERIFIED), batch 1, short context:
>
> decode ceiling ≈ 300e9 B/s ÷ 16.06e9 B/token ≈ **18.7 tokens/s** (≈ 53 ms per token).
>
> It is measured in P2.1 with vLLM: `TODO(run-on: g6.xlarge)` with the command `vllm bench latency --model <8B-instruct> --batch-size 1 --input-len 128 --output-len 128` (see P2.1 `aws.md`). Expect 75–90% of the ceiling: achievable bandwidth is below peak, and kernel launch gaps, the KV reads and sampling add time. P2.1 makes you explain the measured gap term by term.

Two more predictions to make the same way, written in your notes before P2:

1. The same model in **FP8** on the L4 (it supports FP8; the T4 doesn't). The bytes halve, so the ceiling doubles.
2. **Prefill** of a 2,048-token prompt on the L4: FLOPs ≈ 2 × 8e9 × 2048 ≈ 3.3e13. At 121 TFLOP/s (×~0.6 achievable) that is ≈ **0.45 s** TTFT, compute-bound.

## 5. Where the other ops land

| Op | Intensity (fp16) | On an L4 (ridge ≈ 403) |
|---|---|---|
| decode GEMV, batch 1 | ≈ 1 | deep memory-bound |
| decode, batch 64, short context | ≈ 64 | memory-bound |
| prefill GEMM, T = 2048 | ≈ 2048 | compute-bound |
| attention during decode | ≈ 1 (each KV byte is used once per query) | memory-bound at any batch |
| RMSNorm, SiLU, residual add | < 1 | memory-bound: fuse them into neighbours (P5.5) |

---

## Walkthrough

```bash
D=course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline
uv run python $D/examples/01_gpu_specs.py                 # loads + validates gpu_specs.yaml, prints ridge table
uv run python $D/examples/02_gpu_roofline.py              # results/gpu_roofline.png with ops placed
uv run --extra torch python $D/examples/03_measure_bw.py  # T2 only: achieved bandwidth vs spec
```

## What you should see

- `01`: the ridge table above, with a warning line for every `UNVERIFIED` entry and every `null`.
- `02`: `results/gpu_roofline.png`, with one roofline per GPU and the decode/prefill/attention points.
- `03` (T2): `| size | copy GB/s | % of spec |`. `TODO(run-on: g6.xlarge)`. The expected shape is 80–90% of spec for large buffers and much less for small ones (launch overhead, L2-resident data).

## Exercises

```bash
uv run pytest course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/exercises
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [**Verify `gpu_specs.yaml`**](exercises/01-verify-specs/README.md) | easy (and mandatory) | schema test: no entry may claim `VERIFIED` without a document + page |
| 2 | [Ridge points and bound classification](exercises/02-classify/README.md) | easy | fixture GPUs |
| 3 | [The `pct_of_peak` helper](exercises/03-pct-of-peak/README.md) | easy | unit tests; used by every later bench |
| 4 | [Measure achieved bandwidth](exercises/04-measure-bw/README.md) | hard (T2) | ≥ 60% of spec on a real GPU; skipped without CUDA |

## Common mistakes

- **Using sparse TFLOPS** as the peak. You'll think you're at 30% of peak when you're at 60%.
- **Using FP32 CUDA-core FLOPs** for a tensor-core GEMM, or the other way round.
- **Measuring bandwidth with a buffer that fits in L2**: 48 MB on an L4.
- **Comparing GPUs by TFLOPS for a decode workload.** Decode cares about GB/s (and GB of capacity).

## Go deeper

- Modular handbook: *GPU architecture fundamentals* (SMs, threads/warps/blocks, GPU memory, tensor cores), with its interactives, and *Choosing the right GPU* (GPUTable).
- NVIDIA architecture whitepapers: Turing (T4), Ampere (A100), Ada Lovelace (L4, L40S), Hopper (H100), Blackwell. Search nvidia.com for "<arch> architecture whitepaper" and cite the exact document.
- Silicon to Scale ch. 3a/3b (GPU fundamentals and roadmap), ch. 4 (memory hierarchy), Appendix A (hardware reference).
- Horace He, *Making Deep Learning Go Brrrr From First Principles*: compute vs memory vs overhead.

## Go down when…

You want to know *how* warps hide latency and what coalescing means → [P5.1](../../P5-cuda-deep-dive/P5.1-execution-model/README.md), [P5.2](../../P5-cuda-deep-dive/P5.2-memory-coalescing-and-shared-memory/README.md). **Next:** [P1.5 Project: capacity calculator](../P1.5-project-capacity-calculator/README.md).

## Animation

[`animations/roofline.html`](../../../animations/roofline.html): pick a GPU preset. Its numbers match `gpu_specs.yaml` and are equally UNVERIFIED.
