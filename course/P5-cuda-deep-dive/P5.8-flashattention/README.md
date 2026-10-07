# P5.8: FlashAttention v1 → v2 forward (D4)

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) (sm_75+ for the CUDA-core kernels. sm_80+ for flash-attn and the tensor-core stretch. See [aws.md](aws.md)) |
| **Time** | ≈35 min reading + ≈14 h hands-on |
| **Prerequisites** | P5.5 (online softmax), P5.6–P5.7 (tiling, tensor cores), P1.2 (attention in the model) |
| **Reading** | FA-1 (Dao et al. 2022, arXiv 2205.14135), FA-2 (Dao 2023, 2307.08691), FA-3 overview (2407.08608) · tspeterkim/flash-attention-minimal (`flash.cu`) · Modular handbook: *FlashAttention* |
| **D4** | `platform/kernels/include/d4/attention.cuh` |

## Learning objectives

1. Count naive attention's HBM traffic and see why it's memory-bound at long sequence lengths.
2. Derive FlashAttention: tile over K/V, keep online-softmax statistics, never write the N×N matrix.
3. Explain FA-2's changes: loop order (Q outer), state in registers, one rescale per tile, better work split, causal tile skipping.
4. Write decode attention over a **paged** KV cache (q_len = 1), the kernel serving engines run every step.
5. Reach the **L6 exit check**: FA-2 forward ≥ 5× naive at sequence length 4096.

---

## 1. Why naive attention is slow

`S = QKᵀ` is B·H·N² values. At N = 4096 and B·H = 16, that's 268M fp32 values = **1 GiB**. It's written, read and written by softmax, then read again for `P·V`: about 4 GiB of HBM traffic for a computation whose inputs and output total 24 MiB. The FLOPs are fine. The bytes are not.

## 2. FlashAttention

Process K/V in tiles. For each query row, keep running statistics `(m, l)` (max and normalizer) and an **unnormalized** output `o`. For each tile:

```
s_j   = q · k_j / √d                       (tile of scores, on-chip only)
m_new = max(m, max_j s_j)
o     = o · e^{m − m_new} + Σ_j e^{s_j − m_new} · v_j
l     = l · e^{m − m_new} + Σ_j e^{s_j − m_new}
m     = m_new
```

At the end, `out = o / l`. It's P5.5's online-softmax merge applied to (o, l) pairs. The animation [`p5-flashattention.html`](../../../animations/p5-flashattention.html) shows a Q block staying put while K/V blocks stream past, with m, l and o updating per tile.

| | FA-1 | FA-2 |
|---|---|---|
| loop order | outer K/V tiles, inner Q tiles | **outer Q tiles (one per block)**, inner K/V |
| (m, l, o) between tiles | written to HBM | **registers** |
| rescale | per tile, in HBM | per tile, in registers. Fewer non-matmul FLOPs |
| parallelism | over batch × heads | + over query tiles (fills the GPU at small batch) |
| causal | mask | mask + **skip** fully masked tiles (~2× less work) |

`d4/attention.cuh` has all three: naive, the FA-1 structure (one launch per K/V tile, to make the HBM round-trips explicit), and FA-2. They're CUDA-core kernels with fp16 storage and fp32 math. FA-2 with tensor-core matmuls is the stretch goal. FA-3 (Hopper) adds warp specialization and asynchrony (overview only).

## 3. Decode is different

At decode, each sequence has **one** new query and a long KV history. There's no Q tile to amortize over, so the job is to read the KV cache at full bandwidth: memory-bound, just like P1.2 said. And the cache is **paged** (P2.3): a block table maps logical positions to physical blocks. `paged_decode_attention` gives each (sequence, head) a block, splits tokens across warps, and merges their softmax states. That's exercise 4, and it's what P6.3's engine calls.

> **Predict first.** At N = 4096, B·H = 16, fp16: bytes moved by naive (Q, K, V, S×3 passes, P, O) vs FA-2 (Q, K, V read once per query tile, O written once). The ratio is roughly the speedup you'd expect if both were bandwidth-bound. Then measure.

---

## Walkthrough

See [aws.md](aws.md): `attention_bench`, the four exercise tests, `vs_flash_attn.py`, and ncu on naive vs FA-2 (`dram__bytes.sum`).

## What you should see

- `attention_bench`: naive slowest, FA-1 in between, FA-2 fastest. At N = 4096, FA-2 ≥ 5× naive (exit check). Causal FA-2 ≈ 2× faster than full.
- ncu: naive moves gigabytes, FA-2 tens of megabytes.
- `vs_flash_attn.py` on an L4: flash-attn (tensor cores) well ahead of our CUDA-core FA-2. That gap is the tensor-core stretch.

`TODO(run-on: g4dn.xlarge)` (and `g6.xlarge` for flash-attn)

## Bench

| N | causal | naive ms | FA-1 ms | FA-2 ms | FA-2 / naive | SDPA ms | flash-attn ms |
|---|---|---|---|---|---|---|---|
| 1024 / 2048 / 4096 | no / yes | | | | (exit: ≥ 5× at 4096) | | `TODO(run-on: g6.xlarge)` |

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Naive attention vs a reference](exercises/01-naive/README.md) | `p5.8_01-naive` |
| 2 | [FA-1 structure (after annotating flash-attention-minimal)](exercises/02-fa1/README.md) | `p5.8_02-fa1` |
| 3 | [FA-2 forward + causal (L6 exit check)](exercises/03-fa2/README.md) | `p5.8_03-fa2` + `attention_bench` |
| 4 | *(hard)* [Decode attention over a paged KV layout](exercises/04-paged-decode/README.md) | `p5.8_04-paged-decode` |

## Common mistakes

- Rescaling `o` by `e^{m−m_new}` but forgetting `l` (or the reverse).
- NaN from `exp(-inf − (-inf))` on fully masked tiles or rows. Guard `m_new == -inf`.
- Causal masking with `j ≥ i` instead of `j > i` (the diagonal must be visible).
- Comparing a CUDA-core FA-2 with flash-attn and concluding the algorithm is slow. It's the tensor cores.

## Go deeper

- FA-1/FA-2/FA-3 papers. Dao-AILab/flash-attention (`csrc/flash_attn/`, CUTLASS-based).
- vLLM's paged attention kernels (`csrc/attention/`) at the pinned SHA: the production version of exercise 4.
- LeetGPU L6 attention problems (Lane B).

**Next:** [P5.9 Triton](../P5.9-triton/README.md).
